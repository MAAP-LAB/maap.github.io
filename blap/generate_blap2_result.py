import torch
import torch.nn as nn
from transformers import BertConfig
from blap.model.BLAP2.QFormer import BertLMHeadModel, BertLayer, BertAttention, BertSelfAttention
from blap.config.BLAP2_Config import BLAP2_Stage2_Config
import json

class AdapterIntegratedQFormer(BertLMHeadModel):
    """Enhanced Q-Former with adapter integration points"""
    
    def __init__(self, config):
        super().__init__(config)
        self.adapter_points = {}
        self._register_adapter_hooks()
    
    def _register_adapter_hooks(self):
        """Register hooks to measure I/O at key adapter integration points"""
        adapter_layers = [0, 2, 4, 6, 8, 10]  # Every 2nd layer for cross-attention
        
        for i, layer in enumerate(self.bert.encoder.layer):
            if i in adapter_layers and layer.has_cross_attention:
                # Hook before cross-attention
                layer.crossattention.self.register_forward_hook(
                    self._create_measurement_hook(f"layer_{i}_cross_attn_input")
                )
                # Hook after cross-attention output
                layer.crossattention.output.register_forward_hook(
                    self._create_measurement_hook(f"layer_{i}_cross_attn_output")
                )
    
    def _create_measurement_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            self.adapter_points[name] = {
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'device': tensor.device,
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item()
            }
        return hook

def measure_qformer_layers():
    """Comprehensively measure Q-Former layer dimensions and adapter integration points"""
    
    # Load configuration
    config_path = "C:/Users/hyoun/maap.github.io/blap/blap/checkpoint/config.json"
    blap2_config = BLAP2_Stage2_Config.from_file(config_path)
    
    # Extract parameters
    num_query_tokens = blap2_config.num_query_tokens
    audio_embed_dim = blap2_config.audio_encoder.embed_dim_audio
    
    # Configure Q-Former
    qformer_config = BertConfig.from_pretrained("bert-base-uncased")
    qformer_config.encoder_width = audio_embed_dim  # 1408 for CLaMP3 integration
    qformer_config.add_cross_attention = True
    qformer_config.cross_attention_freq = 2
    qformer_config.query_length = num_query_tokens
    
    # Create enhanced Q-Former
    qformer = AdapterIntegratedQFormer(qformer_config)
    
    print("="*80)
    print("Q-FORMER LAYER DIMENSION ANALYSIS FOR ADAPTER INTEGRATION")
    print("="*80)
    
    print(f"\nConfiguration Summary:")
    print(f"  - Hidden Size: {qformer_config.hidden_size}")
    print(f"  - Intermediate Size: {qformer_config.intermediate_size}")
    print(f"  - Number of Layers: {qformer_config.num_hidden_layers}")
    print(f"  - Number of Attention Heads: {qformer_config.num_attention_heads}")
    print(f"  - Audio Encoder Width: {audio_embed_dim}")
    print(f"  - Query Tokens: {num_query_tokens}")
    print(f"  - Cross-Attention Frequency: Every {qformer_config.cross_attention_freq} layers")
    
    # Detailed layer analysis
    adapter_integration_points = []
    
    print(f"\n{'='*60}")
    print("DETAILED LAYER-BY-LAYER ANALYSIS")
    print(f"{'='*60}")
    
    for i, layer in enumerate(qformer.bert.encoder.layer):
        print(f"\n🔍 LAYER {i}:")
        print("-" * 40)
        
        # Self-Attention dimensions
        print("  📌 Self-Attention:")
        print(f"    Query/Key/Value: {layer.attention.self.query.in_features} → {layer.attention.self.query.out_features}")
        print(f"    Output Dense: {layer.attention.output.dense.in_features} → {layer.attention.output.dense.out_features}")
        
        # Cross-Attention dimensions (key for adapter integration)
        if layer.has_cross_attention:
            print("  🎯 Cross-Attention (ADAPTER INTEGRATION POINT):")
            print(f"    Query: {layer.crossattention.self.query.in_features} → {layer.crossattention.self.query.out_features}")
            print(f"    Key/Value: {layer.crossattention.self.key.in_features} → {layer.crossattention.self.key.out_features}")
            print(f"    Output Dense: {layer.crossattention.output.dense.in_features} → {layer.crossattention.output.dense.out_features}")
            
            adapter_integration_points.append({
                'layer': i,
                'type': 'cross_attention',
                'input_dim': layer.crossattention.self.key.in_features,  # From encoder (CLaMP3)
                'hidden_dim': layer.crossattention.self.query.in_features,  # Q-Former hidden
                'output_dim': layer.crossattention.output.dense.out_features
            })
        
        # Feed-Forward dimensions
        print("  🔧 Feed-Forward Networks:")
        print(f"    Intermediate: {layer.intermediate.dense.in_features} → {layer.intermediate.dense.out_features}")
        print(f"    Output: {layer.output.dense.in_features} → {layer.output.dense.out_features}")
        
        if hasattr(layer, 'intermediate_query'):
            print(f"    Query Intermediate: {layer.intermediate_query.dense.in_features} → {layer.intermediate_query.dense.out_features}")
            print(f"    Query Output: {layer.output_query.dense.in_features} → {layer.output_query.dense.out_features}")
    
    # Adapter recommendations
    print(f"\n{'='*60}")
    print("ADAPTER INTEGRATION RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print("\n🎯 OPTIMAL ADAPTER PLACEMENT POINTS:")
    for point in adapter_integration_points:
        print(f"  Layer {point['layer']}: CLaMP3({point['input_dim']}) → Adapter → Q-Former({point['hidden_dim']})")
    
    print(f"\n📊 ADAPTER ARCHITECTURE RECOMMENDATIONS:")
    print(f"  1. Input Adapter: CLaMP3 features (768) → Q-Former encoder_width ({audio_embed_dim})")
    print(f"  2. Cross-Attention Adapters: At layers {[p['layer'] for p in adapter_integration_points]}")
    print(f"  3. Bottleneck Dimension: Recommend {qformer_config.hidden_size // 4} for efficiency")
    
    # Create dummy inputs to measure runtime dimensions
    print(f"\n{'='*60}")
    print("RUNTIME DIMENSION MEASUREMENT")
    print(f"{'='*60}")
    
    batch_size = 2
    seq_length = 100
    
    # Simulate CLaMP3 features
    clamp3_features = torch.randn(batch_size, seq_length, 768)  # CLaMP3 output dim
    query_tokens = torch.randn(batch_size, num_query_tokens, qformer_config.hidden_size)
    
    print(f"Test Input Shapes:")
    print(f"  CLaMP3 Features: {clamp3_features.shape}")
    print(f"  Query Tokens: {query_tokens.shape}")
    
    # Test forward pass (without full computation)
    try:
        with torch.no_grad():
            # Simulate adapter transformation
            adapted_features = torch.randn(batch_size, seq_length, audio_embed_dim)
            attention_mask = torch.ones(batch_size, seq_length)
            
            output = qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=adapted_features,
                encoder_attention_mask=attention_mask,
                return_dict=True
            )
            
            print(f"\nOutput Shapes:")
            print(f"  Last Hidden State: {output.last_hidden_state.shape}")
            print(f"  Query Features: {output.last_hidden_state[:, :num_query_tokens, :].shape}")
            
            # Print adapter measurement points
            if qformer.adapter_points:
                print(f"\n🔍 Adapter Integration Point Measurements:")
                for name, stats in qformer.adapter_points.items():
                    print(f"  {name}: {stats['shape']} | μ={stats['mean']:.4f} σ={stats['std']:.4f}")
    
    except Exception as e:
        print(f"Runtime measurement failed: {e}")
    
    return {
        'config': qformer_config,
        'adapter_points': adapter_integration_points,
        'clamp3_input_dim': 768,
        'qformer_encoder_width': audio_embed_dim,
        'hidden_size': qformer_config.hidden_size,
        'num_query_tokens': num_query_tokens
    }

def generate_adapter_code_template(measurements):
    """Generate adapter implementation template based on measurements"""
    
    template = f'''
# Generated Adapter Template for CLaMP3-BLAP Q-Former Integration
import torch
import torch.nn as nn

class CLaMP3_QFormer_Adapter(nn.Module):
    """Adapter for integrating CLaMP3 features with BLAP Q-Former"""
    
    def __init__(self, 
                 clamp3_dim={measurements['clamp3_input_dim']}, 
                 qformer_dim={measurements['qformer_encoder_width']},
                 bottleneck_dim={measurements['hidden_size'] // 4}):
        super().__init__()
        
        # Bottleneck adapter architecture
        self.down_project = nn.Linear(clamp3_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(bottleneck_dim, qformer_dim)
        self.layer_norm = nn.LayerNorm(qformer_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, clamp3_features):
        # clamp3_features: (batch, seq_len, {measurements['clamp3_input_dim']})
        x = self.down_project(clamp3_features)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        x = self.layer_norm(x)
        return x  # (batch, seq_len, {measurements['qformer_encoder_width']})

# Integration points for Q-Former layers:
ADAPTER_LAYERS = {[p['layer'] for p in measurements['adapter_points']]}
CROSS_ATTENTION_LAYERS = {[p['layer'] for p in measurements['adapter_points'] if p['type'] == 'cross_attention']}
'''
    
    return template

if __name__ == "__main__":
    measurements = measure_qformer_layers()
    
    # Generate adapter template
    adapter_template = generate_adapter_code_template(measurements)
    
    print(f"\n{'='*60}")
    print("GENERATED ADAPTER TEMPLATE")
    print(f"{'='*60}")
    print(adapter_template)
    
    # Save measurements to JSON
    output_file = "qformer_adapter_measurements.json"
    with open(output_file, 'w') as f:
        # Convert non-serializable objects
        serializable_measurements = {
            'clamp3_input_dim': measurements['clamp3_input_dim'],
            'qformer_encoder_width': measurements['qformer_encoder_width'],
            'hidden_size': measurements['hidden_size'],
            'num_query_tokens': measurements['num_query_tokens'],
            'adapter_points': measurements['adapter_points']
        }
        json.dump(serializable_measurements, f, indent=2)
    
    print(f"\n✅ Measurements saved to: {output_file}")
    print("🚀 Ready for adapter implementation!")