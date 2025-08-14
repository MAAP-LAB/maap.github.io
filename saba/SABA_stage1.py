"""
Stage 1: CLaMP3 + Adapter + QFormer Integration
목표: CLaMP3의 limitation 해결 - audio와 symbolic을 text를 bridge로 연결

Architecture:
1. CLaMP3: Audio/Symbolic feature extraction
2. Adapter: Audio-specific bottleneck adapter for feature alignment
3. QFormer: 32 query embedding 생성을 위한 cross-attention

Key Features:
- Audio-text, Symbolic-text contrastive learning
- Adapter는 세 가지 목적 함수에 대한 손실 적용
- QFormer의 cross-attention으로 modality 간 연결
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import random
import json
import sys

# Add project paths
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/clamp3/code")
sys.path.append(PROJECT_ROOT + "/blap")

from transformers import BertConfig, AutoTokenizer, BertTokenizer


from blap.blap.config.BLAP2_Config import BLAP2_Stage1_Config
from blap.blap.config.BLAP2_Config import BLAP2_Stage2_Config

from clamp3.code.utils import CLaMP3Model
from clamp3.code.config import *

from saba.qformer import BertModel, BertLMHeadModel
from saba.adapter import BottleneckAdapter
from saba.base_models import SABABase
from saba.SABA_outputs import SABAStage1Output


class SABA_Stage1(SABABase):
    """
    Stage 1: CLaMP3 + Adapter + QFormer Integration
    
    This model combines:
    1. CLaMP3 for audio/symbolic feature extraction
    2. Bottleneck adapters for feature alignment 
    3. QFormer for cross-modal understanding with 32 query embeddings
    """
    
    def __init__(self,
                 clamp3_weights_path: str = None,
                 blap_checkpoint_path: str = None,
                 num_query_tokens: int = 32,
                 bottleneck_dim: int = 128,
                 dropout: float = 0.1,
                 embed_dim: int = 256):
        
        world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
        local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

        if world_size > 1:
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            dist.init_process_group(backend='nccl')
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        if CLAMP3_DETERMINISTIC:
            seed = 42 + global_rank
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # SABABase 초기화
        super().__init__()
        
        self.num_query_tokens = num_query_tokens
        self.bottleneck_dim = bottleneck_dim
        self.embed_dim = embed_dim
        
        audio_config = BertConfig(vocab_size=1,
                            hidden_size=AUDIO_HIDDEN_SIZE,
                            num_hidden_layers=AUDIO_NUM_LAYERS,
                            num_attention_heads=AUDIO_HIDDEN_SIZE//64,
                            intermediate_size=AUDIO_HIDDEN_SIZE*4,
                            max_position_embeddings=MAX_AUDIO_LENGTH)
        symbolic_config = BertConfig(vocab_size=1,
                                    hidden_size=M3_HIDDEN_SIZE,
                                    num_hidden_layers=PATCH_NUM_LAYERS,
                                    num_attention_heads=M3_HIDDEN_SIZE//64,
                                    intermediate_size=M3_HIDDEN_SIZE*4,
                                    max_position_embeddings=PATCH_LENGTH)
        clamp3 = CLaMP3Model(audio_config=audio_config,
                            symbolic_config=symbolic_config,
                            global_rank=global_rank,
                            world_size=world_size,
                            text_model_name=TEXT_MODEL_NAME,
                            hidden_size=CLAMP3_HIDDEN_SIZE,
                            load_m3=CLAMP3_LOAD_M3)
        clamp3 = clamp3.to(self.device)
        
        
        
        # Freeze CLaMP3
        for param in self.clamp3.parameters():
            param.requires_grad = False
        
        # 2. Initialize QFormer with BLAP checkpoint
        self.qformer, self.query_tokens = self.initQformer(
            num_query_token=num_query_tokens,
            audio_width=1024,  # BLAP uses 1024
            cross_attention_freq=2,
            checkpoint_path=blap_checkpoint_path
        )
        
        # 3. Tokenizer setup
        self.tokenizer = self.init_tokenizer()
        self.qformer.resize_token_embeddings(len(self.tokenizer))
        
        # 4. Adapter
        self.adapter = BottleneckAdapter(
            clamp3_dim=CLAMP3_HIDDEN_SIZE,
            qformer_dim=self.qformer.config.hidden_size,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout
        )

        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.to(self.device)
        
        print(f"✅ Stage1 Model initialized with {num_query_tokens} query tokens")
        print(f"   Device: {self.device}")
        print(f"   Total parameters: {self.get_parameter_count():,}")
    
    def forward(self, 
                text: List[str],
                audio_features: torch.Tensor = None,
                audio_masks: torch.Tensor = None,
                symbolic_inputs: torch.Tensor = None,
                symbolic_masks: torch.Tensor = None,
                modality: str = "audio") -> SABAStage1Output:
        """
        Forward pass with 3 objectives for adapter loss
        """
        batch_size = len(text)
        
        # 1. Process text
        text_tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        
        text_output = self.qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        
        # 3. Process modality features
        if modality == "audio" and audio_features is not None:
            # Get CLaMP3 audio features
            clamp3_features = self.clamp3.get_audio_features(
                audio_features, audio_masks, get_global=False
            )
            modality_masks = audio_masks
        elif modality == "symbolic" and symbolic_inputs is not None:
            # Get CLaMP3 symbolic features
            clamp3_features = self.clamp3.get_symbolic_features(
                symbolic_inputs, symbolic_masks, get_global=False
            )
            modality_masks = symbolic_masks
        else:
            raise ValueError(f"Invalid modality {modality} or missing inputs")
            
        # 4. Apply universal adapter
        adapted_features = self.adapter(clamp3_features)
        
        # 5. Get query embeddings through QFormer cross-attention
        query_embeddings = self.saba_qformer.get_query_embeddings(
            encoder_hidden_states=adapted_features,
            encoder_attention_mask=modality_masks
        )
        
        ###============== Image-text Contrastive ===================###

        ###============== Image-text Matching ===================###

        ##================= Image Captioning ========================##
        
        
    
    def extract_query_embeddings(self, 
                                 audio_features: torch.Tensor = None,
                                 audio_masks: torch.Tensor = None,
                                 symbolic_inputs: torch.Tensor = None,
                                 symbolic_masks: torch.Tensor = None,
                                 modality: str = "audio") -> torch.Tensor:
        """
        Extract 32 query embeddings for downstream tasks
        
        Returns:
            Query embeddings [batch, num_query_tokens, hidden_size]
        """
        with torch.no_grad():
            if modality == "audio" and audio_features is not None:
                # Get adapted audio features
                clamp3_audio_features = self.clamp3.get_audio_features(
                    audio_features, audio_masks, get_global=False
                )
                adapted_audio_features = self.adapter(clamp3_audio_features)
                
                # QFormer cross-attention
                return self.saba_qformer.get_query_embeddings(
                    encoder_hidden_states=adapted_audio_features,
                    encoder_attention_mask=audio_masks
                )
            
            elif modality == "symbolic" and symbolic_inputs is not None:
                # Get adapted symbolic features
                clamp3_symbolic_features = self.clamp3.get_symbolic_features(
                    symbolic_inputs, symbolic_masks, get_global=False
                )
                adapted_symbolic_features = self.adapter(clamp3_symbolic_features)
                
                # QFormer cross-attention
                return self.saba_qformer.get_query_embeddings(
                    encoder_hidden_states=adapted_symbolic_features,
                    encoder_attention_mask=symbolic_masks
                )
            
            else:
                raise ValueError(f"Invalid modality {modality} or missing inputs")


# Usage example and testing functions
def create_stage1_model(clamp3_weights_path: str = None, 
                       blap_checkpoint_path: str = None) -> SABA_Stage1:
    """Create and initialize Stage 1 model"""
    model = SABA_Stage1(
        clamp3_weights_path=clamp3_weights_path,
        blap_checkpoint_path=blap_checkpoint_path,
        num_query_tokens=32,
        hidden_size=768,
        bottleneck_dim=128,
        dropout=0.1
    )
    
    # For Stage 1 training: freeze CLaMP3, train adapters and QFormer
    model.freeze_clamp3()
    
    print(f"Model created with {model.get_trainable_parameters():,} trainable parameters")
    return model


def test_stage1_forward():
    """Test Stage 1 model forward pass"""
    model = create_stage1_model()
    model.eval()
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 64
    feature_dim = 768
    
    text = ["A classical piano piece", "Jazz music with saxophone"]
    audio_features = torch.randn(batch_size, seq_len, feature_dim)
    audio_masks = torch.ones(batch_size, seq_len)
    
    # Test audio modality
    with torch.no_grad():
        outputs = model(
            text=text,
            audio_features=audio_features,
            audio_masks=audio_masks,
            modality="audio"
        )
    
    print("✅ Stage 1 forward pass successful!")
    print(f"Total loss: {outputs['total_loss'].item():.4f}")
    print(f"Audio-text loss: {outputs['audio_text_loss'].item():.4f}")
    print(f"Query features shape: {outputs['audio_query_features'].shape}")
    
    # Test query embedding extraction
    query_embeddings = model.extract_query_embeddings(
        audio_features=audio_features,
        audio_masks=audio_masks,
        modality="audio"
    )
    print(f"Query embeddings shape: {query_embeddings.shape}")  # [batch, 32, 768]


if __name__ == "__main__":
    print("🚀 Testing Stage 1: CLaMP3 + Adapter + QFormer")
    test_stage1_forward()
    print("✅ All tests passed!")