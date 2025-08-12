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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

# Add project paths
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/clamp3/code")
sys.path.append(PROJECT_ROOT + "/blap")

from transformers import BertConfig, AutoTokenizer, BertTokenizer
from clamp3.code.utils import CLaMP3Model
from clamp3.code.config import *
from saba.qformer import BertModel
from adapter import BottleneckAdapter


class BottleneckAdapter(nn.Module):
    """Audio-specific bottleneck adapter for feature alignment"""
    
    def __init__(self, 
                 clamp3_dim: int = 768,
                 qformer_dim: int = 1024, 
                 bottleneck_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        # Bottleneck architecture
        self.adapter = BottleneckAdapter()

        # Residual connection projection
        self.projection = self.adapter.projection(clamp3_dim, qformer_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(qformer_dim)
    
    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with bottleneck transformation
        
        Args:
            x: Input features [batch, seq_len, clamp3_dim]
            residual: Optional residual connection [batch, seq_len, qformer_dim]
        
        Returns:
            Adapted features [batch, seq_len, qformer_dim]
        """
        # Bottleneck transformation
        x = self.adapter.projection(x.shape[1], 1024)
        adapted = self.adapter.sque(x)
        
        # Residual connection
        if residual is None:
            residual = self.projection(x)
        
        # Combine with residual and normalize
        output = self.layer_norm(adapted + residual)
        
        return output


class SABA_Stage1(nn.Module):
    """
    Stage 1: CLaMP3 + Adapter + QFormer Integration
    
    This model combines:
    1. CLaMP3 for audio/symbolic feature extraction
    2. Bottleneck adapters for feature alignment 
    3. QFormer for cross-modal understanding with 32 query embeddings
    """
    
    def __init__(self,
                 clamp3_weights_path: str = None,
                 audio_config: BertConfig = None,
                 symbolic_config: BertConfig = None,
                 qformer_config: BertConfig = None,
                 num_query_tokens: int = 32,
                 hidden_size: int = 768,
                 bottleneck_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_query_tokens = num_query_tokens
        self.hidden_size = hidden_size
        
        # Initialize configurations if not provided
        if audio_config is None:
            audio_config = self._create_audio_config()
        if symbolic_config is None:
            symbolic_config = self._create_symbolic_config()
        if qformer_config is None:
            qformer_config = self._create_qformer_config()
        
        # 1. CLaMP3 Model - for audio/symbolic feature extraction
        self.clamp3 = CLaMP3Model(
            audio_config=audio_config,
            symbolic_config=symbolic_config,
            text_model_name=TEXT_MODEL_NAME,
            hidden_size=CLAMP3_HIDDEN_SIZE,
            load_m3=CLAMP3_LOAD_M3
        )
        
        # Load CLaMP3 weights if provided
        if clamp3_weights_path and Path(clamp3_weights_path).exists():
            self._load_clamp3_weights(clamp3_weights_path)
        
        # 2. Single Universal Bottleneck Adapter (for both audio and symbolic)
        self.adapter = BottleneckAdapter(
            clamp3_dim=CLAMP3_HIDDEN_SIZE,
            qformer_dim=qformer_config.hidden_size,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout
        )
        
        # 4. QFormer - for cross-attention and query embedding generation
        self.qformer = BertModel(qformer_config, add_pooling_layer=False)
        
        # 5. Learnable Query Tokens (32 embeddings)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, qformer_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)
        
        # 6. Text tokenizer for text processing
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        
        # 7. Projection layers for different objectives
        self.text_proj = nn.Linear(qformer_config.hidden_size, hidden_size)
        self.audio_proj = nn.Linear(qformer_config.hidden_size, hidden_size)
        self.symbolic_proj = nn.Linear(qformer_config.hidden_size, hidden_size)
        
        # 8. Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        print(f"✅ Stage1 Model initialized with {num_query_tokens} query tokens")
    
    def _create_audio_config(self) -> BertConfig:
        """Create audio encoder configuration"""
        return BertConfig(
            vocab_size=1,
            hidden_size=AUDIO_HIDDEN_SIZE,
            num_hidden_layers=AUDIO_NUM_LAYERS,
            num_attention_heads=AUDIO_HIDDEN_SIZE//64,
            intermediate_size=AUDIO_HIDDEN_SIZE*4,
            max_position_embeddings=MAX_AUDIO_LENGTH
        )
    
    def _create_symbolic_config(self) -> BertConfig:
        """Create symbolic encoder configuration"""
        return BertConfig(
            vocab_size=1,
            hidden_size=M3_HIDDEN_SIZE,
            num_hidden_layers=PATCH_NUM_LAYERS,
            num_attention_heads=M3_HIDDEN_SIZE//64,
            intermediate_size=M3_HIDDEN_SIZE*4,
            max_position_embeddings=PATCH_LENGTH
        )
    
    def _create_qformer_config(self) -> BertConfig:
        """Create QFormer configuration with cross-attention"""
        config = BertConfig.from_pretrained("bert-base-uncased")
        config.encoder_width = self.hidden_size  # Encoder width for cross-attention
        config.add_cross_attention = True
        config.cross_attention_freq = 2  # Add cross-attention every 2 layers
        config.query_length = self.num_query_tokens
        return config
    
    def _load_clamp3_weights(self, weights_path: str):
        """Load CLaMP3 pre-trained weights"""
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)
            
            # Handle module prefix
            if next(iter(state_dict)).startswith('module.'):
                state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            
            self.clamp3.load_state_dict(state_dict, strict=False)
            print(f"✅ CLaMP3 weights loaded from {weights_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load CLaMP3 weights: {e}")
    
    def freeze_clamp3(self):
        """Freeze CLaMP3 parameters for adapter-only training"""
        for param in self.clamp3.parameters():
            param.requires_grad = False
        print("🧊 CLaMP3 frozen - only adapters trainable")
    
    def freeze_qformer(self):
        """Freeze QFormer parameters"""
        for param in self.qformer.parameters():
            param.requires_grad = False
        self.query_tokens.requires_grad = False
        print("🧊 QFormer frozen")
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def process_text(self, texts: List[str], max_length: int = MAX_TEXT_LENGTH) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process text inputs"""
        encoded = self.tokenizer( # Bert Tokenizer
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encoded.input_ids.to(self.device), encoded.attention_mask.to(self.device)
    
    def compute_contrastive_loss(self, 
                                query_features: torch.Tensor, 
                                target_features: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss between query and target features"""
        # Normalize features
        query_features = F.normalize(query_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(query_features, target_features.T) / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        batch_size = query_features.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Compute cross-entropy loss
        loss_q2t = F.cross_entropy(logits, labels)
        loss_t2q = F.cross_entropy(logits.T, labels)
        
        return (loss_q2t + loss_t2q) / 2
    
    def forward(self, 
                text: List[str],
                audio_features: torch.Tensor = None,
                audio_masks: torch.Tensor = None,
                symbolic_inputs: torch.Tensor = None,
                symbolic_masks: torch.Tensor = None,
                modality: str = "audio") -> Dict[str, torch.Tensor]:
        """
        Forward pass for Stage 1 training
        
        Args:
            text: List of text descriptions
            audio_features: Audio feature inputs [batch, seq_len, feat_dim] 
            audio_masks: Audio attention masks [batch, seq_len]
            symbolic_inputs: Symbolic music inputs [batch, seq_len, patch_size]
            symbolic_masks: Symbolic attention masks [batch, seq_len]
            modality: "audio" or "symbolic" or "both"
        
        Returns:
            Dictionary with losses and features

        TODO: making three loss(adapted_feature came from clamp3 to adapter, text as input data in Qformer) of blip2. 
        """
        batch_size = len(text)
        
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        
        # 3. Process audio modality if provided
        if modality in ["audio"] and audio_features is not None:
            # Get CLaMP3 audio features (local features, not global)
            clamp3_feature = self.clamp3.get_audio_features(
                audio_features, audio_masks, get_global=False
            )
        elif modality in ["symbolic"] and audio_features is not None:

            clamp3_feature = self.clamp3.get_symbolic_features(
                symbolic_inputs, symbolic_masks, get_global=False
            )
            
        # Apply universal adapter
        adapted_features = self.adapter(clamp3_feature)
        adapted_features_atts = torch.ones(adapted_features.size()[:-1], dtype=torch.long).to(
        self.device
        )
        
        # QFormer cross-attention with adapted audio features
        query_tokens = self.query_tokens.expand(adapted_features.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=adapted_features,
            encoder_attention_mask=adapted_features_atts,
            use_cache=True,
            return_dict=True,
        )

        outputs = {}
        total_loss = 0.0


        """
        Wrtie code about three loss, which should be itc, itm, lm, and then return output
        """
        
        return outputs
    
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
                batch_size = audio_features.shape[0]
                query_tokens = self.query_tokens.expand(batch_size, -1, -1)
                
                # Get adapted audio features
                clamp3_audio_features = self.clamp3.get_audio_features(
                    audio_features, audio_masks, get_global=False
                )
                adapted_audio_features = self.adapter(clamp3_audio_features)
                
                # QFormer cross-attention
                query_output = self.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=adapted_audio_features,
                    encoder_attention_mask=audio_masks,
                    return_dict=True
                )
                
                return query_output.last_hidden_state[:, :self.num_query_tokens, :]
            
            elif modality == "symbolic" and symbolic_inputs is not None:
                batch_size = symbolic_inputs.shape[0]
                query_tokens = self.query_tokens.expand(batch_size, -1, -1)
                
                # Get adapted symbolic features
                clamp3_symbolic_features = self.clamp3.get_symbolic_features(
                    symbolic_inputs, symbolic_masks, get_global=False
                )
                adapted_symbolic_features = self.adapter(clamp3_symbolic_features)
                
                # QFormer cross-attention
                query_output = self.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=adapted_symbolic_features,
                    encoder_attention_mask=symbolic_masks,
                    return_dict=True
                )
                
                return query_output.last_hidden_state[:, :self.num_query_tokens, :]
            
            else:
                raise ValueError(f"Invalid modality {modality} or missing inputs")


# Usage example and testing functions
def create_stage1_model(clamp3_weights_path: str = None) -> SABA_Stage1:
    """Create and initialize Stage 1 model"""
    model = SABA_Stage1(
        clamp3_weights_path=clamp3_weights_path,
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