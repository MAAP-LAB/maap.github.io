"""
SABA Q-Former 클래스
BLAP2_Pretrain과 BLAP2Output을 참고하여 작성된 SABA용 QFormer

Based on BLAP2 implementation and BLIP2 architecture
Copyright references to original BLIP2 implementation by Salesforce
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import sys
import numpy as np

# Add project paths
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/blap")

from transformers import BertConfig, BertTokenizer
from saba.qformer import BertLMHeadModel, BertModel
from saba.adapter import BottleneckAdapter
from saba.SABA_outputs import SABAOutput, SABAIntermediateOutput, SABASimilarity
from saba.base_models import SABABase

from clamp3.inference.clamp3_score import calculate_pairwise_similarity

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """

    if not is_dist_avail_and_initialized():
        return tensors
    
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


class SABA_QFormer(SABABase):
    """
    SABA용 Q-Former 클래스
    BLAP2의 QFormer 초기화 방식을 참고하여 구현
    
    Features:
    - BLAP checkpoint에서 가중치 로드 지원
    - Audio-Text 및 Symbolic-Text contrastive learning
    - Cross-attention을 통한 query embedding 생성
    """
    
    def __init__(self, 
                 blap_checkpoint_path: Optional[str] = None,
                 config: Optional[str] = None,
                 num_query_tokens: int = 32,
                 audio_width: int = 1024,
                 cross_attention_freq: int = 2,
                 embed_dim: int = 256,
                 max_txt_len: int = 120,
                 bottleneck_dim: int = 128,
                 dropout: float = 0.1,
                 device: str = "cpu"):
        """
        SABA Q-Former 초기화
        
        Args:
            blap_checkpoint_path: BLAP checkpoint 경로 (옵션)
            config: Q-Former 설정 (옵션, 기본값 사용)
            num_query_tokens: Query token 개수
            audio_width: Audio encoder 출력 차원 (adapter 출력)
            cross_attention_freq: Cross-attention 빈도
            embed_dim: Projection 출력 차원
            max_txt_len: 최대 텍스트 길이
            device: 모델을 로드할 디바이스
        """
        super().__init__()
        
        self.device = device
        self.num_query_tokens = num_query_tokens
        self.audio_width = audio_width
        self.embed_dim = embed_dim
        self.max_txt_len = max_txt_len
        
        # Tokenizer 초기화 (BLAP2 방식)
        self.tokenizer = self._init_tokenizer()
        
        # QFormer 초기화 (BLAP2 방식)
        self.qformer, self.query_tokens = self.init_Qformer(blap_checkpoint_path, config)
        self.tokenizer = self.init_tokenizer()
        
        # Resize token embeddings
        self.qformer.resize_token_embeddings(len(self.tokenizer))
        
        # Query weights 복사 (BLAP2 방식)
        self._copy_query_weights()

        # Projection layers (BLAP2 방식)
        self.audio_proj = self._create_projection(self.qformer.config.hidden_size, embed_dim)
        self.text_proj = self._create_projection(self.qformer.config.hidden_size, embed_dim)
        self.symbolic_proj = self._create_projection(self.qformer.config.hidden_size, embed_dim)

        # Audio-Text Matching head (for Loss 2)
        self.atm_head = nn.Linear(self.qformer.config.hidden_size, 2)
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # Bottleneck Adapter (integrated into QFormer)
        self.adapter = BottleneckAdapter(
            clamp3_dim=audio_width,  # CLaMP3 output dimension
            qformer_dim=self.qformer.config.hidden_size,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout
        )
        
        self.to(device)
        
        print(f"✅ SABA Q-Former initialized")
        print(f"   Device: {device}")
        print(f"   Query tokens: {num_query_tokens}")
        print(f"   Audio width: {audio_width}")
        print(f"   Parameters: {self.get_parameter_count():,}")
        print(f"   BLAP weights loaded: {blap_checkpoint_path is not None}")
    
    
    def get_audio_embeddings(self, audio_embeds, audio_atts=None):
        """
        Audio embeddings를 QFormer를 통해 처리
        BLAP2의 get_audioEmbeddings 방식을 참고
        
        Args:
            audio_embeds: Audio embeddings [batch_size, seq_len, embed_dim]
            audio_atts: Audio attention masks [batch_size, seq_len]
            
        Returns:
            Audio features [batch_size, num_query_tokens, embed_dim]
        """
        if audio_atts is None:
            audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(self.device)
        
        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)
        
        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            use_cache=True,
            return_dict=True,
        )
        
        audio_feats = F.normalize(
            self.audio_proj(query_output.last_hidden_state), dim=-1
        )
        
        return audio_feats
    
    def get_text_embeddings(self, texts):
        """
        Text embeddings를 QFormer를 통해 처리
        BLAP2의 get_captionEmbeddings 방식을 참고
        
        Args:
            texts: List of text strings
            
        Returns:
            Text features [batch_size, embed_dim]
        """
        text_tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        
        text_output = self.qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        
        # Take CLS token as text embedding
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        
        return text_feat
    
    def get_query_embeddings(self, 
                           encoder_hidden_states: torch.Tensor,
                           encoder_attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Query embeddings를 추출합니다.
        
        Args:
            encoder_hidden_states: 인코더 hidden states (CLaMP3 output) [batch_size, seq_len, hidden_size]
            encoder_attention_mask: 인코더 attention mask [batch_size, seq_len]
            
        Returns:
            Query embeddings [batch_size, num_query_tokens, hidden_size]
        """
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.size()[:-1], dtype=torch.long
            ).to(self.device)
        
        # Apply adapter to CLaMP3 features
        adapted_features = self.adapter(encoder_hidden_states)
        
        query_tokens = self.query_tokens.expand(encoder_hidden_states.shape[0], -1, -1)
        
        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=adapted_features,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            return_dict=True,
        )
        
        return query_output.last_hidden_state[:, :self.num_query_tokens, :]
    
    def forward(self, 
                texts: List[str],
                audio_embeds: torch.Tensor = None,
                audio_atts: torch.Tensor = None,
                symbolic_embeds: torch.Tensor = None,
                symbolic_atts: torch.Tensor = None,
                modality: str = "audio") -> SABAOutput:
        """
        SABA QFormer forward pass with clear data flow:
        - Audio/Symbolic: CLaMP3 embeddings → Adapter → QFormer
        - Text: texts → BERT tokenizer → QFormer
        
        Args:
            texts: List of text descriptions
            audio_embeds: CLaMP3 audio embeddings [batch_size, seq_len, clamp3_dim]
            audio_atts: Audio attention masks [batch_size, seq_len]
            symbolic_embeds: CLaMP3 symbolic embeddings [batch_size, seq_len, clamp3_dim]
            symbolic_atts: Symbolic attention masks [batch_size, seq_len]
            modality: "audio" or "symbolic"
            
        Returns:
            SABAOutput with 3 losses and features
        """
        
        # Determine batch size and device
        if modality == "audio" and audio_embeds is not None:
            batch_size = audio_embeds.shape[0]
            raw_embeds = audio_embeds
            modality_atts = audio_atts
        elif modality == "symbolic" and symbolic_embeds is not None:
            batch_size = symbolic_embeds.shape[0]
            raw_embeds = symbolic_embeds
            modality_atts = symbolic_atts
        else:
            raise ValueError(f"Invalid modality '{modality}' or missing embeddings")
        
        device = raw_embeds.device
        
        # Create attention masks if not provided
        if modality_atts is None:
            modality_atts = torch.ones(raw_embeds.size()[:-1], dtype=torch.long).to(device)
        
        print(f"📊 Forward pass - {modality} modality:")
        print(f"   Input embeds shape: {raw_embeds.shape}")
        print(f"   Batch size: {batch_size}")
        
        # ============== STEP 1: Audio/Symbolic Processing ==============
        # CLaMP3 embeddings → Adapter → adapted features
        print("🔄 Step 1: Applying adapter to CLaMP3 features...")
        modality_embeds = self.adapter(raw_embeds)  # [batch, seq_len, qformer_dim]
        print(f"   After adapter shape: {modality_embeds.shape}")
        
        # Query tokens for cross-attention
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)  # [batch, 32, hidden]
        
        # QFormer cross-attention: query tokens attend to adapted features
        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=modality_embeds,
            encoder_attention_mask=modality_atts,
            use_cache=True,
            return_dict=True,
        )
        
        # Extract query embeddings and project for contrastive learning
        query_features = query_output.last_hidden_state[:, :self.num_query_tokens, :]  # [batch, 32, hidden]
        
        if modality == "audio":
            modality_feats = F.normalize(
                self.audio_proj(query_features), dim=-1
            )  # [batch, 32, embed_dim]
        else:  # symbolic
            modality_feats = F.normalize(
                self.symbolic_proj(query_features), dim=-1
            )  # [batch, 32, embed_dim]
        
        print(f"   Query features shape: {query_features.shape}")
        print(f"   Projected features shape: {modality_feats.shape}")
        
        # ============== STEP 2: Text Processing ==============
        # texts → BERT tokenizer → QFormer → text features
        print("🔄 Step 2: Processing text through BERT tokenizer...")
        text_tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        
        print(f"   Text tokens shape: {text_tokens.input_ids.shape}")
        
        # Text through QFormer BERT
        text_output = self.qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        
        # Extract [CLS] token and project
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )  # [batch, embed_dim]
        
        print(f"   Text features shape: {text_feat.shape}")
        
        # ============== STEP 3: Three Loss Calculations ==============
        print("🔄 Step 3: Computing 3 losses...")
        
        # Gather features from all GPUs for contrastive learning
        modality_feats_all = concat_all_gather(modality_feats)  # [batch*gpu, 32, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch*gpu, embed_dim]

        ###============== Loss 1: Audio-text Contrastive ===================###
        print("   💡 Loss 1: Contrastive Learning")
        
        # Query-to-text similarity: [batch, batch*gpu, 32]
        sim_q2t = torch.matmul(
            modality_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        
        # Aggregate across query tokens (max pooling)
        sim_m2t, _ = sim_q2t.max(-1)  # [batch, batch*gpu]
        sim_m2t = sim_m2t / self.temperature

        # Text-to-query similarity: [batch, batch*gpu, 32]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), 
            modality_feats_all.permute(0, 2, 1)
        ).squeeze()

        # Aggregate across query tokens
        sim_t2m, _ = sim_t2q.max(-1)  # [batch, batch*gpu]
        sim_t2m = sim_t2m / self.temperature

        # Create targets for contrastive loss
        rank = dist.get_rank() if is_dist_avail_and_initialized() else 0
        targets = torch.linspace(
            rank * batch_size, rank * batch_size + batch_size - 1, 
            batch_size, dtype=int
        ).to(device)
        
        # Contrastive loss
        loss_atc = (
            F.cross_entropy(sim_m2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2m, targets, label_smoothing=0.1)
        ) / 2
        
        print(f"      Contrastive loss: {loss_atc.item():.4f}")

        ###============== Loss 2: Audio-text Matching ===================###
        print("   💡 Loss 2: Matching (Hard Negative Mining)")
        
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        modality_embeds_world = all_gather_with_grad(modality_embeds)
        
        with torch.no_grad():
            # Mask out diagonal for negative sampling
            sim_t2m[:, rank * batch_size : rank * batch_size + batch_size].fill_diagonal_(-10000)
            sim_m2t[:, rank * batch_size : rank * batch_size + batch_size].fill_diagonal_(-10000)            
                
            weights_t2m = F.softmax(sim_t2m, dim=1)
            weights_m2t = F.softmax(sim_m2t, dim=1)

        # Select negative samples using hard negative mining
        modality_embeds_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_t2m[b], 1).item()
            modality_embeds_neg.append(modality_embeds_world[neg_idx])
        modality_embeds_neg = torch.stack(modality_embeds_neg, dim=0)

        text_ids_neg = []
        text_atts_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_m2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # Create positive and negative pairs
        text_ids_all = torch.cat([
            text_tokens.input_ids,    # positive
            text_tokens.input_ids,    # positive
            text_ids_neg              # negative
        ], dim=0)
        
        text_atts_all = torch.cat([
            text_tokens.attention_mask,
            text_tokens.attention_mask, 
            text_atts_neg
        ], dim=0)

        # Expand query tokens for all pairs
        query_tokens_matching = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_matching = torch.ones(
            query_tokens_matching.size()[:-1], dtype=torch.long
        ).to(device)
        attention_mask_all = torch.cat([query_atts_matching, text_atts_all], dim=1)

        # Create modality embeddings for all pairs (pos, neg, pos)
        modality_embeds_all = torch.cat([
            modality_embeds,         # positive
            modality_embeds_neg,     # negative  
            modality_embeds          # positive
        ], dim=0)
        
        modality_atts_all = torch.ones(
            modality_embeds_all.size()[:-1], dtype=torch.long
        ).to(device)

        # Forward through QFormer for matching
        output_matching = self.qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_matching,
            attention_mask=attention_mask_all,
            encoder_hidden_states=modality_embeds_all,
            encoder_attention_mask=modality_atts_all,
            return_dict=True,
        )

        # Extract query representations and classify
        query_embeddings_matching = output_matching.last_hidden_state[:, :self.num_query_tokens, :]
        matching_logits = self.atm_head(query_embeddings_matching).mean(dim=1)  # [3*batch, 2]

        # Labels: [1, 1, ..., 1, 0, 0, ..., 0] (positive, positive, negative)
        matching_labels = torch.cat([
            torch.ones(batch_size, dtype=torch.long),   # positive
            torch.zeros(2 * batch_size, dtype=torch.long)  # negative + positive
        ], dim=0).to(device)
        
        loss_atm = F.cross_entropy(matching_logits, matching_labels)
        print(f"      Matching loss: {loss_atm.item():.4f}")

        ##================= Audio Captioning (Loss 3) ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        # Total loss combining all three objectives
        total_loss = loss_atc + loss_atm + loss_lm
        
        # Create similarity object for outputs
        sims = SABASimilarity(
            sim_a2t=sim_a2t if modality == "audio" else None,
            sim_t2a=sim_t2a if modality == "audio" else None,
            sim_s2t=sim_a2t if modality == "symbolic" else None,
            sim_t2s=sim_t2a if modality == "symbolic" else None,
            sim_a2t_targets=targets if modality == "audio" else None,
            sim_t2a_targets=targets if modality == "audio" else None,
        )
        
        intermediate_output = SABAIntermediateOutput(
            audio_embeds=modality_embeds if modality == "audio" else None,
            audio_features=modality_feats if modality == "audio" else None,
            symbolic_embeds=modality_embeds if modality == "symbolic" else None,
            symbolic_features=modality_feats if modality == "symbolic" else None,
            text_embeds=text_feat,
            text_features=text_feat,
            query_embeddings=query_output.last_hidden_state[:, :self.num_query_tokens, :],
            adapted_features=modality_embeds,
        )
        
        return SABAOutput(
            loss=total_loss,
            loss_atc=loss_atc if modality == "audio" else None,
            loss_stc=loss_atc if modality == "symbolic" else None,
            loss_atm=loss_atm,
            loss_lm=loss_lm,
            sims=sims,
            intermediate_output=intermediate_output
        )
    
    def freeze_qformer_only(self):
        """QFormer만 동결하고 projection layers와 adapter는 학습 가능하게 유지"""
        for param in self.qformer.parameters():
            param.requires_grad = False
        self.query_tokens.requires_grad = True
        
        # Keep adapter trainable
        for param in self.adapter.parameters():
            param.requires_grad = True
            
        print("🧊 Q-Former frozen (projections and adapter trainable)")
    
    def freeze_adapter(self):
        """Adapter만 동결"""
        for param in self.adapter.parameters():
            param.requires_grad = False
        print("🧊 Adapter frozen")
    
    def unfreeze_adapter(self):
        """Adapter 해제"""
        for param in self.adapter.parameters():
            param.requires_grad = True
        print("🔥 Adapter unfrozen")
    
    def get_adapter_parameters(self):
        """Adapter 파라미터만 반환"""
        return self.adapter.parameters()
    
    def get_trainable_adapter_parameters(self):
        """훈련 가능한 Adapter 파라미터 수 반환"""
        return sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)


# Training utility functions
def train_adapter_only(saba_qformer: SABA_QFormer):
    """Adapter만 훈련하도록 설정"""
    saba_qformer.freeze()  # 모든 파라미터 동결
    saba_qformer.unfreeze_adapter()  # Adapter만 해제
    print("🎯 Training mode: Adapter only")

def train_adapter_and_qformer(saba_qformer: SABA_QFormer):
    """Adapter와 QFormer를 함께 훈련하도록 설정"""
    saba_qformer.unfreeze()  # 모든 파라미터 해제
    print("🎯 Training mode: Adapter + QFormer")

def train_qformer_only(saba_qformer: SABA_QFormer):
    """QFormer만 훈련하도록 설정 (Adapter 동결)"""
    saba_qformer.unfreeze()  # 모든 파라미터 해제
    saba_qformer.freeze_adapter()  # Adapter만 동결
    print("🎯 Training mode: QFormer only")

