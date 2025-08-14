"""
SABA Output Classes
BLAP2Outputs를 참고하여 SABA용 출력 클래스 정의
"""

from dataclasses import dataclass
from typing import Optional
import torch
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)


@dataclass
class SABASimilarity(ModelOutput):
    """Audio-Text similarity scores"""
    sim_a2t: torch.FloatTensor = None  # Audio to Text similarity
    sim_t2a: torch.FloatTensor = None  # Text to Audio similarity
    
    sim_s2t: Optional[torch.FloatTensor] = None  # Symbolic to Text similarity
    sim_t2s: Optional[torch.FloatTensor] = None  # Text to Symbolic similarity
    
    # Momentum encoders (if used)
    sim_a2t_m: Optional[torch.FloatTensor] = None
    sim_t2a_m: Optional[torch.FloatTensor] = None
    
    # Targets for contrastive learning
    sim_a2t_targets: Optional[torch.FloatTensor] = None
    sim_t2a_targets: Optional[torch.FloatTensor] = None


@dataclass
class SABAIntermediateOutput(ModelOutput):
    """
    Intermediate outputs for SABA models
    
    Args:
        audio_embeds: Audio embeddings from encoder [batch_size, seq_len, embed_dim]
        audio_features: Audio features from QFormer [batch_size, num_query_tokens, embed_dim]
        symbolic_embeds: Symbolic embeddings from encoder [batch_size, seq_len, embed_dim]
        symbolic_features: Symbolic features from QFormer [batch_size, num_query_tokens, embed_dim]
        text_embeds: Text embeddings [batch_size, embed_dim]
        text_features: Text features [batch_size, embed_dim]
        
        # Cross-modal outputs
        query_embeddings: Query embeddings from QFormer [batch_size, num_query_tokens, hidden_size]
        adapted_features: Features after adapter [batch_size, seq_len, adapter_dim]
        
        # Training outputs
        atm_logits: Audio-Text Matching logits [batch_size*3, 2]
        atm_labels: Audio-Text Matching labels [batch_size*3]
        vl_embedding: Vision-Language embeddings for ITM
        
        # Decoder outputs
        decoder_output: Output from decoder (if applicable)
        decoder_labels: Labels for language modeling
    """
    
    # Uni-modal features
    audio_embeds: Optional[torch.FloatTensor] = None
    audio_features: Optional[torch.FloatTensor] = None
    symbolic_embeds: Optional[torch.FloatTensor] = None
    symbolic_features: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    text_features: Optional[torch.FloatTensor] = None
    
    # Cross-modal features
    query_embeddings: Optional[torch.FloatTensor] = None
    adapted_features: Optional[torch.FloatTensor] = None
    
    # Momentum encoders (if used)
    audio_embeds_m: Optional[torch.FloatTensor] = None
    text_embeds_m: Optional[torch.FloatTensor] = None
    
    # Multimodal encoder outputs
    encoder_output: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    encoder_output_neg: Optional[BaseModelOutputWithPoolingAndCrossAttentions] = None
    
    # Audio-Text Matching
    atm_logits: Optional[torch.FloatTensor] = None
    atm_labels: Optional[torch.LongTensor] = None
    vl_embedding: Optional[torch.FloatTensor] = None
    
    # Multimodal decoder outputs
    decoder_output: Optional[CausalLMOutputWithCrossAttentions] = None
    decoder_labels: Optional[torch.LongTensor] = None


@dataclass
class SABAOutput(ModelOutput):
    """
    Main output class for SABA models
    
    Args:
        loss: Total loss (combination of all losses)
        loss_atc: Audio-Text Contrastive loss
        loss_stc: Symbolic-Text Contrastive loss (if applicable)
        loss_atm: Audio-Text Matching loss
        loss_lm: Language Modeling loss
        
        sims: Similarity scores
        intermediate_output: Intermediate outputs and features
    """
    
    # Losses
    loss: Optional[torch.FloatTensor] = None
    loss_atc: Optional[torch.FloatTensor] = None  # Audio-Text Contrastive
    loss_stc: Optional[torch.FloatTensor] = None  # Symbolic-Text Contrastive
    loss_atm: Optional[torch.FloatTensor] = None  # Audio-Text Matching
    loss_lm: Optional[torch.FloatTensor] = None   # Language Modeling
    
    # Similarity scores
    sims: Optional[SABASimilarity] = None
    
    # Intermediate outputs
    intermediate_output: Optional[SABAIntermediateOutput] = None


@dataclass
class SABAStage1Output(ModelOutput):
    """Output for SABA Stage 1 (CLaMP3 + Adapter + QFormer)"""
    
    loss: Optional[torch.FloatTensor] = None
    loss_contrastive: Optional[torch.FloatTensor] = None
    
    # Features
    audio_features: Optional[torch.FloatTensor] = None
    symbolic_features: Optional[torch.FloatTensor] = None
    text_features: Optional[torch.FloatTensor] = None
    query_embeddings: Optional[torch.FloatTensor] = None
    adapted_features: Optional[torch.FloatTensor] = None
    
    # Similarity
    sims: Optional[SABASimilarity] = None


@dataclass
class SABAStage2Output(ModelOutput):
    """Output for SABA Stage 2 (QFormer + T5)"""
    
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    
    # Generated text
    generated_text: Optional[list] = None
    
    # Features
    query_embeddings: Optional[torch.FloatTensor] = None
    projected_queries: Optional[torch.FloatTensor] = None
    
    # T5 inputs/outputs
    t5_inputs: Optional[dict] = None
    t5_outputs: Optional[CausalLMOutputWithCrossAttentions] = None