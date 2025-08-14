"""
Stage 2: QFormer + Flan-T5 Integration
목표: Stage1에서 학습된 QFormer와 Flan-T5를 통합하여 Audio task 수행

Architecture:
1. Stage1 Model: Pre-trained CLaMP3 + Adapter + QFormer (frozen)
2. QFormer: Stage1에서 pre-trained된 QFormer 활용
3. Flan-T5: Text generation을 위한 T5 모델
4. Audio Task: Music QA, Captioning, Analysis

Key Features:
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
from saba.base_models import SABABase

# Add project paths
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/clamp3/code")
sys.path.append(PROJECT_ROOT + "/blap")

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, BertConfig
from saba.SABA_stage1 import SABA_Stage1
from saba.qformer import BertLMHeadModel, BertModel
from saba.SABA_outputs import SABAStage2Output

class SABA_Stage2(SABABase):
    """
    Stage 2: QFormer + Flan-T5 Integration
    
    This model combines:
    1. Pre-trained Stage1 model (frozen CLaMP3 + Adapter + QFormer)
    2. QFormer from Stage1 (can be fine-tuned)
    3. Flan-T5 for text generation
    4. Projection layers for connecting QFormer outputs to T5 inputs
    """
    
    def __init__(self,
                 stage1_model: SABA_Stage1 = None,
                 qformer_config: Dict = None,
                 blap_checkpoint_path: str = None,
                 flan_t5_model: str = "google/flan-t5-large",
                 freeze_stage1: bool = True,
                 freeze_qformer: bool = False,
                 freeze_t5: bool = False,
                 num_query_tokens: int = 32,
                 embed_dim: int = 256,
                 device_name: str = "auto"):
        
        
        # Device 설정
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            
        # SABABase 초기화
        super().__init__(
            qformer_config=qformer_config,
            blap_checkpoint_path=blap_checkpoint_path,
            num_query_tokens=num_query_tokens,
            embed_dim=embed_dim,
            device_name=device_name
        )
        
        # 1. Stage1 model (optional, for compatibility)
        self.stage1 = stage1_model
        if stage1_model and freeze_stage1:
            self._freeze_stage1()
        
        # 2. Flan-T5 model
        self.t5_config = T5Config.from_pretrained(flan_t5_model)
        self.flan_t5 = T5ForConditionalGeneration.from_pretrained(
            flan_t5_model,
            config=self.t5_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.t5_tokenizer = T5Tokenizer.from_pretrained(flan_t5_model)
        
        # Freeze components if requested
        if freeze_qformer:
            self.freeze_qformer()
        if freeze_t5:
            self._freeze_t5()
        
        # 3. Projection layer: QFormer hidden_size -> T5 hidden_size
        qformer_hidden_size = self.qformer.config.hidden_size
        t5_hidden_size = self.t5_config.d_model
        
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
        print(f"   Stage1 frozen: {freeze_stage1}")
        print(f"   QFormer frozen: {freeze_qformer}")
        print(f"   T5 frozen: {freeze_t5}")
    
    def initQformer(self, checkpoint_path: str):
        """Initialize Q-Former and load from BLAP checkpoint"""
        print("🤖 Initializing Q-Former from BLAP checkpoint...")

        # Create Q-Former config
        qformer_config = BertConfig.from_pretrained("bert-base-uncased")
        qformer_config.encoder_width = 1024  # BLAP checkpoint uses 1024, not 1408
        qformer_config.add_cross_attention = True
        qformer_config.cross_attention_freq = 2
        qformer_config.query_length = self.blap_config.num_query_tokens

        self.qformer_config = qformer_config

        # Initialize Q-Former
        self.qformer = BertLMHeadModel(qformer_config)

        # Initialize query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.blap_config.num_query_tokens, qformer_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)

        # Load from BLAP checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Extract state dict
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Filter Q-Former related weights
                qformer_state_dict = {}
                query_token_state = None

                for key, value in state_dict.items():
                    if 'qformer' in key:
                        # Remove prefix
                        new_key = key.replace('qformer.', '')
                        qformer_state_dict[new_key] = value
                    elif 'query_tokens' in key:
                        query_token_state = value

                # Load Q-Former weights
                if qformer_state_dict:
                    missing_keys, unexpected_keys = self.qformer.load_state_dict(qformer_state_dict, strict=False)
                    print(f"✅ Q-Former loaded with {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")

                # Load query tokens
                if query_token_state is not None:
                    self.query_tokens.data.copy_(query_token_state)
                    print("✅ Query tokens loaded successfully")

            except Exception as e:
                print(f"⚠️ Warning: Could not load BLAP checkpoint: {e}")
    
    def extract_audio_features(self,
                              audio_features: torch.Tensor,
                              audio_masks: torch.Tensor) -> torch.Tensor:
        """
        Extract audio features using Stage1 pipeline + QFormer
        
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
        
        # 3. QFormer cross-attention
        query_outputs = self.qformer(
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
        text_tokens = self.tokenizer(
            prompted_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # 4. Get T5 text embeddings
        with torch.no_grad():
            text_embeddings = self.flan_t5.encoder.embed_tokens(
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
        # 1. Extract audio features through Stage1 + QFormer
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
            target_tokens = self.tokenizer(
                target_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Prepare labels (mask padding tokens)
            labels = target_tokens.input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # T5 forward pass
            t5_outputs = self.flan_t5(
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
            generated_tokens = self.flan_t5.generate(
                inputs_embeds=t5_inputs["inputs_embeds"],
                attention_mask=t5_inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # 4. Decode generated tokens
            generated_texts = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            
            return generated_texts
    
    def save_model(self, save_directory: str):
        """Save Stage2 model components"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save QFormer
        qformer_path = save_path / "qformer"
        torch.save({
            "model_state_dict": self.qformer.state_dict(),
            "config": self.qformer.config
        }, qformer_path.with_suffix(".pth"))
        
        # Save T5 model
        t5_path = save_path / "flan_t5"
        self.flan_t5.save_pretrained(t5_path)
        
        # Save projection layer
        torch.save({
            "qformer_to_t5_proj": self.qformer_to_t5_proj.state_dict(),
            "config": {
                "qformer_hidden_size": self.qformer.config.hidden_size,
                "t5_hidden_size": self.t5_config.d_model,
                "num_query_tokens": self.num_query_tokens
            }
        }, save_path / "projection_layer.pth")
        
        print(f"✅ Stage2 model saved to {save_directory}")
    
    def load_model(self, load_directory: str):
        """Load Stage2 model components"""
        load_path = Path(load_directory)
        
        # Load QFormer
        qformer_path = load_path / "qformer.pth"
        if qformer_path.exists():
            checkpoint = torch.load(qformer_path, map_location=self.device)
            self.qformer.load_state_dict(checkpoint["model_state_dict"])
        
        # Load T5 model
        t5_path = load_path / "flan_t5"
        if t5_path.exists():
            self.flan_t5 = T5ForConditionalGeneration.from_pretrained(t5_path)
        
        # Load projection layer
        proj_path = load_path / "projection_layer.pth"
        if proj_path.exists():
            checkpoint = torch.load(proj_path, map_location=self.device)
            self.qformer_to_t5_proj.load_state_dict(checkpoint["qformer_to_t5_proj"])
            
        print(f"✅ Stage2 model loaded from {load_directory}")


# Factory functions
def create_stage2_model(stage1_model: SABA_Stage1,
                       flan_t5_model: str = "google/flan-t5-large",
                       freeze_stage1: bool = True,
                       freeze_qformer: bool = False,
                       freeze_t5: bool = False) -> SABA_Stage2:
    """Create Stage2 model from pre-trained Stage1"""
    
    model = SABA_Stage2(
        stage1_model=stage1_model,
        flan_t5_model=flan_t5_model,
        freeze_stage1=freeze_stage1,
        freeze_qformer=freeze_qformer,
        freeze_t5=freeze_t5
    )
    
    print(f"Stage2 model created with {model.get_trainable_parameters():,} trainable parameters")
    return model


def test_stage2_forward():
    """Test Stage2 model forward pass"""
    from saba.SABA_stage1 import create_stage1_model
    
    # Create Stage1 model
    stage1_model = create_stage1_model()
    
    # Create Stage2 model
    stage2_model = create_stage2_model(stage1_model, "google/flan-t5-base")  # Use base for testing
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
    print("🚀 Testing Stage 2: QFormer + Flan-T5")
    test_stage2_forward()
    print("✅ All tests passed!")