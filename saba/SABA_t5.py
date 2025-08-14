"""
SABA Unified Model: Stage1+2 통합 모델 (Metric 용도)
- forward: contrastive learning for metrics
- generate: text generation for metrics
- 실제 훈련은 SABA_QFormer에서 수행
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add project paths
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/clamp3/code")

from transformers import T5ForConditionalGeneration, T5Tokenizer
from clamp3.code.utils import CLaMP3Model
from clamp3.code.config import *
from saba.SABA_qformer import SABA_QFormer
from saba.base_models import SABABase


class SABA_t5(SABABase):
    """
    SABA 통합 모델 - Metric 평가 및 추론 전용
    실제 훈련은 SABA_QFormer에서 수행
    """
    
    def __init__(self,
                 clamp3_weights_path: str = None,
                 blap_checkpoint_path: str = None,
                 config_path: str = None,
                 flan_t5_model: str = "google/flan-t5-large",
                 device_name: str = "auto"):
        
        super().__init__()
        
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device_name
        
        # 1. CLaMP3 Model (frozen)
        self.clamp3 = CLaMP3Model(
            audio_config=self._create_audio_config(),
            symbolic_config=self._create_symbolic_config(),
            text_model_name=TEXT_MODEL_NAME,
            hidden_size=CLAMP3_HIDDEN_SIZE,
            load_m3=CLAMP3_LOAD_M3
        )
        
        if clamp3_weights_path:
            self._load_clamp3_weights(clamp3_weights_path)
        
        # Freeze CLaMP3
        for param in self.clamp3.parameters():
            param.requires_grad = False
        
        # 2. SABA QFormer (with integrated adapter)
        self.saba_qformer, self.query_tokens = self.init_Qformer(blap_checkpoint_path, config_path)
        
        # 3. T5 for generation (optional)
        self.flan_t5 = None
        self.t5_tokenizer = None
        self.qformer_to_t5_proj = None
        
        if flan_t5_model:
            self._init_t5(flan_t5_model)
        
        self.to(device_name)
        print(f"✅ SABA Unified Model initialized on {device_name}")
        print(f"   CLaMP3: frozen")
        print(f"   SABA_QFormer: with integrated adapter")
        print(f"   T5: {'loaded' if self.flan_t5 else 'not loaded'}")
    
    def _create_audio_config(self):
        from transformers import BertConfig
        return BertConfig(
            vocab_size=1,
            hidden_size=AUDIO_HIDDEN_SIZE,
            num_hidden_layers=AUDIO_NUM_LAYERS,
            num_attention_heads=AUDIO_HIDDEN_SIZE//64,
            intermediate_size=AUDIO_HIDDEN_SIZE*4,
            max_position_embeddings=MAX_AUDIO_LENGTH
        )
    
    def _create_symbolic_config(self):
        from transformers import BertConfig
        return BertConfig(
            vocab_size=1,
            hidden_size=M3_HIDDEN_SIZE,
            num_hidden_layers=PATCH_NUM_LAYERS,
            num_attention_heads=M3_HIDDEN_SIZE//64,
            intermediate_size=M3_HIDDEN_SIZE*4,
            max_position_embeddings=PATCH_LENGTH
        )
    
    def _load_clamp3_weights(self, weights_path: str):
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)
            if next(iter(state_dict)).startswith('module.'):
                state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            self.clamp3.load_state_dict(state_dict, strict=False)
            print("✅ CLaMP3 weights loaded")
        except Exception as e:
            print(f"⚠️ Could not load CLaMP3 weights: {e}")
    
    def _init_t5(self, flan_t5_model: str):
        """Initialize T5 for generation"""
        print("📝 Initializing T5 for generation...")
        
        self.flan_t5 = T5ForConditionalGeneration.from_pretrained(
            flan_t5_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.t5_tokenizer = T5Tokenizer.from_pretrained(flan_t5_model)
        
        # Projection layer
        self.qformer_to_t5_proj = nn.Linear(
            self.saba_qformer.qformer.config.hidden_size,
            self.flan_t5.config.d_model
        )
        
        # Task templates
        self.prompt_templates = {
            "caption": "Describe the music: ",
            "qa": "Question: {question} Answer: ",
            "analysis": "Analyze this music: ",
            "classification": "What genre is this music? ",
            "emotion": "What emotion does this music convey? "
        }
    
    def forward(self, 
                text: List[str],
                audio_features: torch.Tensor = None,
                audio_masks: torch.Tensor = None,
                symbolic_inputs: torch.Tensor = None,
                symbolic_masks: torch.Tensor = None,
                modality: str = "audio"):
        """
        Forward pass for metric evaluation (contrastive learning)
        
        Args:
            text: List of text descriptions
            audio_features: Audio features [batch, seq_len, feat_dim]
            audio_masks: Audio attention masks 
            symbolic_inputs: Symbolic inputs
            symbolic_masks: Symbolic attention masks
            modality: "audio" or "symbolic"
            
        Returns:
            SABA output with losses and features
        """
        # 1. Get CLaMP3 features
        if modality == "audio" and audio_features is not None:
            clamp3_features = self.clamp3.get_audio_features(
                audio_features, audio_masks, get_global=False
            )
            modality_masks = audio_masks
        elif modality == "symbolic" and symbolic_inputs is not None:
            clamp3_features = self.clamp3.get_symbolic_features(
                symbolic_inputs, symbolic_masks, get_global=False
            )
            modality_masks = symbolic_masks
        else:
            raise ValueError(f"Invalid modality {modality} or missing inputs")
        
        # 2. SABA QFormer forward (includes adapter processing)
        return self.saba_qformer.forward(
            audio_embeds=clamp3_features if modality == "audio" else None,
            symbolic_embeds=clamp3_features if modality == "symbolic" else None,
            texts=text,
            audio_atts=modality_masks if modality == "audio" else None,
            symbolic_atts=modality_masks if modality == "symbolic" else None,
            modality=modality
        )
    
    def generate(self,
                audio_features: torch.Tensor = None,
                audio_masks: torch.Tensor = None,
                symbolic_inputs: torch.Tensor = None,
                symbolic_masks: torch.Tensor = None,
                text_inputs: List[str] = None,
                task_type: str = "caption",
                modality: str = "audio",
                max_length: int = 128,
                num_beams: int = 4) -> List[str]:
        """
        Generate text for metric evaluation
        
        Returns:
            List of generated texts
        """
        if self.flan_t5 is None:
            raise ValueError("T5 model not initialized. Please provide flan_t5_model in __init__")
        
        self.eval()
        with torch.no_grad():
            # 1. Get CLaMP3 features
            if modality == "audio" and audio_features is not None:
                clamp3_features = self.clamp3.get_audio_features(
                    audio_features, audio_masks, get_global=False
                )
                modality_masks = audio_masks
            elif modality == "symbolic" and symbolic_inputs is not None:
                clamp3_features = self.clamp3.get_symbolic_features(
                    symbolic_inputs, symbolic_masks, get_global=False
                )
                modality_masks = symbolic_masks
            else:
                raise ValueError(f"Invalid modality {modality} or missing inputs")
            
            # 2. Get query embeddings from SABA QFormer
            query_embeddings = self.saba_qformer.get_query_embeddings(
                encoder_hidden_states=clamp3_features,
                encoder_attention_mask=modality_masks
            )
            
            # 3. Prepare T5 inputs
            t5_inputs = self._prepare_t5_inputs(query_embeddings, text_inputs, task_type)
            
            # 4. Generate
            generated_tokens = self.flan_t5.generate(
                inputs_embeds=t5_inputs["inputs_embeds"],
                attention_mask=t5_inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                pad_token_id=self.t5_tokenizer.pad_token_id,
                eos_token_id=self.t5_tokenizer.eos_token_id
            )
            
            # 5. Decode
            generated_texts = self.t5_tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            
            return generated_texts
    
    def _prepare_t5_inputs(self, query_embeddings, text_inputs, task_type):
        """Prepare T5 inputs for generation"""
        batch_size = query_embeddings.shape[0]
        
        # Project to T5 space
        projected_queries = self.qformer_to_t5_proj(query_embeddings)
        
        # Add task prompts
        if task_type in self.prompt_templates:
            if task_type == "qa":
                prompted_texts = [
                    self.prompt_templates[task_type].format(question=text) 
                    for text in text_inputs
                ]
            else:
                prompted_texts = [
                    self.prompt_templates[task_type] + text 
                    for text in text_inputs
                ]
        else:
            prompted_texts = text_inputs
        
        # Tokenize
        text_tokens = self.t5_tokenizer(
            prompted_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get T5 embeddings
        text_embeddings = self.flan_t5.encoder.embed_tokens(text_tokens.input_ids)
        
        # Combine
        combined_embeddings = torch.cat([projected_queries, text_embeddings], dim=1)
        
        query_mask = torch.ones(
            batch_size, query_embeddings.shape[1],
            device=self.device, dtype=torch.long
        )
        combined_mask = torch.cat([query_mask, text_tokens.attention_mask], dim=1)
        
        return {
            "inputs_embeds": combined_embeddings,
            "attention_mask": combined_mask
        }
    
    def get_metrics(self, outputs):
        """Extract metrics from model outputs"""
        metrics = {}
        
        if hasattr(outputs, 'loss'):
            metrics['total_loss'] = outputs.loss.item()
        
        if hasattr(outputs, 'loss_atc') and outputs.loss_atc is not None:
            metrics['contrastive_loss'] = outputs.loss_atc.item()
        
        if hasattr(outputs, 'loss_atm') and outputs.loss_atm is not None:
            metrics['matching_loss'] = outputs.loss_atm.item()
        
        if hasattr(outputs, 'loss_lm') and outputs.loss_lm is not None:
            metrics['captioning_loss'] = outputs.loss_lm.item()
        
        # Add similarity metrics if available
        if hasattr(outputs, 'sims') and outputs.sims is not None:
            if outputs.sims.sim_a2t is not None:
                metrics['audio_text_similarity'] = outputs.sims.sim_a2t.mean().item()
            if outputs.sims.sim_t2a is not None:
                metrics['text_audio_similarity'] = outputs.sims.sim_t2a.mean().item()
        
        return metrics


def create_saba(clamp3_weights_path: str = None,
                       blap_checkpoint_path: str = None,
                       flan_t5_model: str = "google/flan-t5-large",
                       for_generation: bool = True) -> SABA_Unified:
    """
    Create SABA Unified model
    
    Args:
        for_generation: Whether to load T5 for generation (False for contrastive only)
    """
    return SABA_Unified(
        clamp3_weights_path=clamp3_weights_path,
        blap_checkpoint_path=blap_checkpoint_path,
        flan_t5_model=flan_t5_model if for_generation else None
    )


if __name__ == "__main__":
    # Test contrastive learning
    model = create_saba_unified(
        blap_checkpoint_path="saba/checkpoint.ckpt",
        for_generation=False  # Only contrastive
    )
    
    batch_size = 2
    audio_features = torch.randn(batch_size, 64, 768)
    audio_masks = torch.ones(batch_size, 64)
    text = ["Classical music", "Jazz music"]
    
    # Test forward
    outputs = model.forward(
        text=text,
        audio_features=audio_features,
        audio_masks=audio_masks,
        modality="audio"
    )
    
    # Get metrics
    metrics = model.get_metrics(outputs)
    print("✅ SABA Unified test successful!")
    print(f"Metrics: {metrics}")