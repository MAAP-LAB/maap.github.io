"""
Bottleneck Adapter Architecture
projection -> down -> norm + residual -> up
Integrates sequentially with Q-Former layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension,dimension)
        self.linear_2 = nn.Linear(dimension,dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)

        return swiglu

class BottleneckAdapter(nn.Module):
    """
    Classic bottleneck adapter architecture:
    projection (clamp3_dim -> qformer_dim) -> down -> norm + add -> up
    """
    
    def __init__(self, 
                 clamp3_dim: int = 768,           # CLaMP3 *.npy feature dimension
                 qformer_dim: int = 1024,         # Q-Former encoder_width (from BLAP checkpoint)
                 bottleneck_dim: int = 128,        # Bottleneck dimension (smaller)
                 ckpt_path: str = None,
                 dropout: float = 0.1):
        super().__init__()
        
        self.projection = nn.Linear(clamp3_dim, qformer_dim, bias=False)

        self.bottleneck = nn.Sequential(
              nn.Linear(qformer_dim, bottleneck_dim, bias=False)
            , nn.LayerNorm(bottleneck_dim)
            , SwiGLU(bottleneck_dim)
            , nn.Dropout(dropout)
            , nn.Linear(bottleneck_dim, qformer_dim, bias=False)
        )
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize adapter weights with small random values"""
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.02)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.xavier_uniform_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, clamp3_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through bottleneck adapter
        
        Args:
            clamp3_features: CLaMP3 features (batch, seq, clamp3_dim)
            residual: Residual connection from previous layer (batch, seq, qformer_dim)
            
        Returns:
            Adapted features (batch, seq, qformer_dim)
        """
        # 1. Project CLaMP3 features to Q-Former dimension
        projected = self.projection(clamp3_features)  # (batch, seq, qformer_dim)
        # 2. forwarding Sequentially
        adapted = self.bottleneck(projected)
        # residual block
        output = projected + adapted
        
        return output