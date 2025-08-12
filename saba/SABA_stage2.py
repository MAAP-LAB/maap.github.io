"""
Stage 2: QFormer + Flan-T5 Integration with LoRA
목표: Stage1에서 학습된 QFormer와 Flan-T5를 LoRA로 fine-tuning하여 Audio task 수행

Architecture:
1. Stage1 Model: Pre-trained CLaMP3 + Adapter + QFormer (frozen)
2. QFormer: Fine-tune with LoRA for audio-specific adaptation  
3. Flan-T5: LoRA-based fine-tuning for text generation
4. Audio Task: Music QA, Captioning, Analysis

Key Features:
- LoRA fine-tuning으로 효율적인 adaptation
- Stage1의 32 query embeddings 활용
- Audio-to-text generation pipeline
- Pre-aligning through contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import sys

# Add project paths
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/clamp3/code")
sys.path.append(PROJECT_ROOT + "/blap")

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, BertConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from saba.SABA_stage1 import SABA_Stage1
from saba.qformer import BertLMHeadModel, BertModel


class LoRAQFormer(nn.Module):
    """QFormer with LoRA adaptation for audio-specific fine-tuning"""
    
    def __init__(self, 
                 stage1_qformer: BertModel,
                 lora_config: LoraConfig = None):
        super().__init__()
        
        # Use pre-trained QFormer from Stage1
        self.base_qformer = stage1_qformer
        
        # Default LoRA config for QFormer
        if lora_config is None:
            lora_config = LoraConfig(
                r=16,                # Low-rank dimension
                lora_alpha=32,       # LoRA scaling parameter  
                target_modules=["query", "key", "value", "dense"],  # Target attention layers
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
        
        # Apply LoRA to QFormer
        self.qformer = get_peft_model(self.base_qformer, lora_config)
        
        print(f"✅ LoRA QFormer initialized with rank {lora_config.r}")
    
    def forward(self, **kwargs):
        """Forward pass through LoRA-adapted QFormer"""
        return self.qformer(**kwargs)


class LoRAFlanT5(nn.Module):
    """Flan-T5 with LoRA adaptation for text generation"""
    
    def __init__(self, 
                 model_name: str = "google/flan-t5-large",
                 lora_config: LoraConfig = None):
        super().__init__()
        
        # Load Flan-T5 base model
        self.config = T5Config.from_pretrained(model_name)
        self.base_model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Default LoRA config for Flan-T5
        if lora_config is None:
            lora_config = LoraConfig(
                r=32,                # Higher rank for generative task
                lora_alpha=64,       # Higher scaling for T5
                target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],  # T5 specific modules
                lora_dropout=0.1,
                bias="none", 
                task_type=TaskType.SEQ_2_SEQ_LM
            )
        
        # Apply LoRA to Flan-T5
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        print(f"✅ LoRA Flan-T5 initialized with rank {lora_config.r}")
    
    def forward(self, **kwargs):
        """Forward pass through LoRA-adapted Flan-T5"""
        return self.model(**kwargs)
    
    def generate(self, **kwargs):
        """Text generation with LoRA-adapted model"""
        return self.model.generate(**kwargs)


class SABA_Stage2(nn.Module):
    """
    Stage 2: QFormer + Flan-T5 Integration with LoRA
    
    This model combines:
    1. Pre-trained Stage1 model (frozen CLaMP3 + Adapter + QFormer base)
    2. LoRA-adapted QFormer for audio-specific fine-tuning
    3. LoRA-adapted Flan-T5 for text generation
    4. Projection layers for connecting QFormer outputs to T5 inputs
    """
    
    def __init__(self,
                 stage1_model: SABA_Stage1,
                 flan_t5_model: str = "google/flan-t5-large",
                 qformer_lora_config: LoraConfig = None,
                 t5_lora_config: LoraConfig = None,
                 freeze_stage1: bool = True):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Pre-trained Stage1 model
        self.stage1 = stage1_model
        
        # Freeze Stage1 components for Stage2 training
        if freeze_stage1:
            self._freeze_stage1()
        
        # 2. LoRA-adapted QFormer (separate from Stage1 QFormer)
        self.lora_qformer = LoRAQFormer(
            stage1_qformer=stage1_model.qformer,
            lora_config=qformer_lora_config
        )
        
        # 3. LoRA-adapted Flan-T5
        self.lora_flan_t5 = LoRAFlanT5(
            model_name=flan_t5_model,
            lora_config=t5_lora_config
        )
        
        # 4. Projection layer: QFormer hidden_size -> T5 hidden_size
        qformer_hidden_size = stage1_model.qformer.config.hidden_size
        t5_hidden_size = self.lora_flan_t5.config.d_model
        
        self.qformer_to_t5_proj = nn.Linear(qformer_hidden_size, t5_hidden_size)
        
        # 5. Query tokens from Stage1 (inherit and potentially adapt)
        self.query_tokens = stage1_model.query_tokens  # Reference to Stage1 query tokens
        self.num_query_tokens = stage1_model.num_query_tokens
        
        # 6. Task-specific prompt templates
        self.prompt_templates = {
            "caption": "Describe the music: ",
            "qa": "Question: {question} Answer: ",
            "analysis": "Analyze this music: ",
            "classification": "What genre is this music? ",
            "emotion": "What emotion does this music convey? "
        }
        
        print(f"✅ Stage2 Model initialized")
        print(f"   QFormer hidden size: {qformer_hidden_size}")
        print(f"   T5 hidden size: {t5_hidden_size}")
        print(f"   Query tokens: {self.num_query_tokens}")
    
    def _freeze_stage1(self):
        """Freeze all Stage1 components"""
        for param in self.stage1.parameters():
            param.requires_grad = False
        print("🧊 Stage1 components frozen")
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def extract_audio_features(self,
                              audio_features: torch.Tensor,
                              audio_masks: torch.Tensor) -> torch.Tensor:
        """
        Extract audio features using Stage1 pipeline + LoRA QFormer
        
        Args:
            audio_features: Audio features [batch, seq_len, feat_dim]
            audio_masks: Audio masks [batch, seq_len]
            
        Returns:
            Query embeddings [batch, num_query_tokens, qformer_hidden_size]
        """
        batch_size = audio_features.shape[0]
        
        # 1. Get Stage1 adapted audio features (frozen)
        with torch.no_grad():
            clamp3_audio_features = self.stage1.clamp3.get_audio_features(
                audio_features, audio_masks, get_global=False
            )
            adapted_audio_features = self.stage1.adapter(clamp3_audio_features)
        
        # 2. Query tokens (from Stage1)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # 3. LoRA QFormer cross-attention (trainable)
        query_outputs = self.lora_qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=adapted_audio_features,
            encoder_attention_mask=audio_masks,
            return_dict=True
        )
        
        # Extract query embeddings
        query_embeddings = query_outputs.last_hidden_state[:, :self.num_query_tokens, :]
        
        return query_embeddings
    
    def prepare_t5_inputs(self,
                         query_embeddings: torch.Tensor,
                         text_inputs: List[str],
                         task_type: str = "caption") -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for T5 generation
        
        Args:
            query_embeddings: Query embeddings [batch, num_query_tokens, qformer_hidden]
            text_inputs: Text prompts/questions
            task_type: Task type for prompt template
            
        Returns:
            Dictionary with T5 input tensors
        """
        batch_size = query_embeddings.shape[0]
        
        # 1. Project query embeddings to T5 space
        projected_queries = self.qformer_to_t5_proj(query_embeddings)  # [batch, 32, t5_hidden]
        
        # 2. Prepare text inputs with task-specific prompts
        if task_type in self.prompt_templates:
            if task_type == "qa":
                # For QA, text_inputs contains questions
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
        
        # 3. Tokenize text inputs
        text_tokens = self.lora_flan_t5.tokenizer(
            prompted_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # 4. Get T5 text embeddings
        with torch.no_grad():
            text_embeddings = self.lora_flan_t5.model.encoder.embed_tokens(
                text_tokens.input_ids
            )  # [batch, text_len, t5_hidden]
        
        # 5. Concatenate query embeddings + text embeddings
        combined_embeddings = torch.cat([projected_queries, text_embeddings], dim=1)
        
        # 6. Create combined attention mask
        query_mask = torch.ones(
            batch_size, self.num_query_tokens,
            device=self.device, dtype=torch.long
        )
        combined_mask = torch.cat([query_mask, text_tokens.attention_mask], dim=1)
        
        return {
            "inputs_embeds": combined_embeddings,
            "attention_mask": combined_mask,
            "text_tokens": text_tokens
        }
    
    def forward(self,
               audio_features: torch.Tensor,
               audio_masks: torch.Tensor,
               text_inputs: List[str],
               target_texts: List[str] = None,
               task_type: str = "caption") -> Dict[str, torch.Tensor]:
        """
        Forward pass for Stage2 training/inference
        
        Args:
            audio_features: Audio features [batch, seq_len, feat_dim]
            audio_masks: Audio masks [batch, seq_len]
            text_inputs: Input text prompts/questions
            target_texts: Target texts for training (optional)
            task_type: Task type for prompt template
            
        Returns:
            Dictionary with loss and logits
        """
        # 1. Extract audio features through Stage1 + LoRA QFormer
        query_embeddings = self.extract_audio_features(audio_features, audio_masks)
        
        # 2. Prepare T5 inputs
        t5_inputs = self.prepare_t5_inputs(query_embeddings, text_inputs, task_type)
        
        outputs = {
            "query_embeddings": query_embeddings,
            "projected_queries": self.qformer_to_t5_proj(query_embeddings)
        }
        
        # 3. Training mode: compute loss with target texts
        if target_texts is not None:
            # Tokenize target texts
            target_tokens = self.lora_flan_t5.tokenizer(
                target_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Prepare labels (mask padding tokens)
            labels = target_tokens.input_ids.clone()
            labels[labels == self.lora_flan_t5.tokenizer.pad_token_id] = -100
            
            # T5 forward pass
            t5_outputs = self.lora_flan_t5(
                inputs_embeds=t5_inputs["inputs_embeds"],
                attention_mask=t5_inputs["attention_mask"],
                decoder_attention_mask=target_tokens.attention_mask,
                labels=labels,
                return_dict=True
            )
            
            outputs.update({
                "loss": t5_outputs.loss,
                "logits": t5_outputs.logits,
                "target_tokens": target_tokens
            })
        
        # 4. Inference mode: no loss computation
        else:
            outputs.update({
                "t5_inputs": t5_inputs,
                "loss": None
            })
        
        return outputs
    
    def generate_text(self,
                     audio_features: torch.Tensor,
                     audio_masks: torch.Tensor,
                     text_inputs: List[str],
                     task_type: str = "caption",
                     max_length: int = 128,
                     num_beams: int = 4,
                     temperature: float = 0.7,
                     do_sample: bool = True) -> List[str]:
        """
        Generate text from audio features
        
        Args:
            audio_features: Audio features [batch, seq_len, feat_dim]
            audio_masks: Audio masks [batch, seq_len]  
            text_inputs: Input prompts/questions
            task_type: Task type for prompt template
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            List of generated texts
        """
        self.eval()
        with torch.no_grad():
            # 1. Extract audio features
            query_embeddings = self.extract_audio_features(audio_features, audio_masks)
            
            # 2. Prepare T5 inputs
            t5_inputs = self.prepare_t5_inputs(query_embeddings, text_inputs, task_type)
            
            # 3. Generate with T5
            generated_tokens = self.lora_flan_t5.generate(
                inputs_embeds=t5_inputs["inputs_embeds"],
                attention_mask=t5_inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.lora_flan_t5.tokenizer.pad_token_id,
                eos_token_id=self.lora_flan_t5.tokenizer.eos_token_id
            )
            
            # 4. Decode generated tokens
            generated_texts = self.lora_flan_t5.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            
            return generated_texts
    
    def save_lora_adapters(self, save_directory: str):
        """Save LoRA adapters"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save QFormer LoRA
        qformer_path = save_path / "qformer_lora"
        self.lora_qformer.qformer.save_pretrained(qformer_path)
        
        # Save T5 LoRA  
        t5_path = save_path / "flan_t5_lora"
        self.lora_flan_t5.model.save_pretrained(t5_path)
        
        # Save projection layer
        torch.save({
            "qformer_to_t5_proj": self.qformer_to_t5_proj.state_dict(),
            "config": {
                "qformer_hidden_size": self.lora_qformer.qformer.config.hidden_size,
                "t5_hidden_size": self.lora_flan_t5.config.d_model,
                "num_query_tokens": self.num_query_tokens
            }
        }, save_path / "projection_layer.pth")
        
        print(f"✅ LoRA adapters saved to {save_directory}")
    
    def load_lora_adapters(self, load_directory: str):
        """Load LoRA adapters"""
        load_path = Path(load_directory)
        
        # Load QFormer LoRA
        qformer_path = load_path / "qformer_lora"
        if qformer_path.exists():
            self.lora_qformer.qformer = PeftModel.from_pretrained(
                self.lora_qformer.base_qformer, qformer_path
            )
        
        # Load T5 LoRA
        t5_path = load_path / "flan_t5_lora" 
        if t5_path.exists():
            self.lora_flan_t5.model = PeftModel.from_pretrained(
                self.lora_flan_t5.base_model, t5_path
            )
        
        # Load projection layer
        proj_path = load_path / "projection_layer.pth"
        if proj_path.exists():
            checkpoint = torch.load(proj_path, map_location=self.device)
            self.qformer_to_t5_proj.load_state_dict(checkpoint["qformer_to_t5_proj"])
            
        print(f"✅ LoRA adapters loaded from {load_directory}")


# Factory functions
def create_stage2_model(stage1_model: SABA_Stage1,
                       flan_t5_model: str = "google/flan-t5-large") -> SABA_Stage2:
    """Create Stage2 model from pre-trained Stage1"""
    
    # Custom LoRA configs
    qformer_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    
    t5_lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    
    model = SABA_Stage2(
        stage1_model=stage1_model,
        flan_t5_model=flan_t5_model,
        qformer_lora_config=qformer_lora_config,
        t5_lora_config=t5_lora_config,
        freeze_stage1=True
    )
    
    print(f"Stage2 model created with {model.get_trainable_parameters():,} trainable parameters")
    return model


def test_stage2_forward():
    """Test Stage2 model forward pass"""
    from saba.SABA_stage1 import create_stage1_model
    
    # Create Stage1 model
    stage1_model = create_stage1_model()
    
    # Create Stage2 model
    stage2_model = create_stage2_model(stage1_model, "google/flan-t5-large")  # Use base for testing
    stage2_model.eval()
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 64
    feature_dim = 768
    
    audio_features = torch.randn(batch_size, seq_len, feature_dim)
    audio_masks = torch.ones(batch_size, seq_len)
    text_inputs = ["Describe this music", "What instruments are playing?"]
    target_texts = ["Classical piano with strings", "Piano and violin ensemble"]
    
    # Test forward pass
    with torch.no_grad():
        outputs = stage2_model(
            audio_features=audio_features,
            audio_masks=audio_masks,
            text_inputs=text_inputs,
            target_texts=target_texts,
            task_type="caption"
        )
    
    print("✅ Stage2 forward pass successful!")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Query embeddings shape: {outputs['query_embeddings'].shape}")
    
    # Test text generation
    generated_texts = stage2_model.generate_text(
        audio_features=audio_features,
        audio_masks=audio_masks,
        text_inputs=text_inputs,
        task_type="caption",
        max_length=32
    )
    
    print("Generated texts:")
    for i, text in enumerate(generated_texts):
        print(f"  {i+1}: {text}")


if __name__ == "__main__":
    print("🚀 Testing Stage 2: QFormer + Flan-T5 with LoRA")
    test_stage2_forward()
    print("✅ All tests passed!")