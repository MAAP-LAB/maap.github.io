"""
Simple BLAP Adapter Trainer with single nn.Linear projection
Loads BLAP checkpoint and trains only projection adapters for cross-attention layers
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import argparse
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

# Add paths
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/clamp3/code")
sys.path.append(PROJECT_ROOT + "/blap")

from transformers import BertConfig, T5TokenizerFast, T5Config, BertTokenizer
from blap.model.BLAP2.modeling_t5 import T5ForConditionalGeneration
from blap.model.BLAP2.QFormer import BertLMHeadModel
from blap.config.BLAP2_Config import BLAP2_Stage2_Config
from clamp3.code.utils import CLaMP3Model
from clamp3.code.config import *
from Adapters.bottleneck_adapter import BottleneckAdapter


class MusicQADataset(Dataset):
    """Dataset for MusicQA with *.npy files"""
    
    def __init__(self, json_path: str, npy_base_path: str = f"{PROJECT_ROOT}/Assets/npys"):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.npy_base_path = Path(npy_base_path)
        
        # Filter data to only include .npy files
        self.data = [item for item in self.data if item['audio_name'].endswith('.npy')]
        
        print(f"Loaded {len(self.data)} samples with .npy audio files")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        audio_name = item['audio_name']
        
        # Construct full path to .npy file
        audio_path = self.npy_base_path / audio_name
        
        # Handle conversation format
        conversation = item['conversation']
        question = conversation[0]['value'] if conversation else "Describe the audio"
        answer = conversation[1]['value'] if len(conversation) > 1 else "Unknown"
        
        return {
            'audio_path': str(audio_path),
            'question': question,
            'answer': answer
        }


class SimpleBLAPAdapterTrainer(nn.Module):
    """
    Simple BLAP trainer with checkpoint loading and single nn.Linear projection adapters
    """
    
    def __init__(self, 
                 blap_checkpoint_path: str,
                 clamp3_weights_path: str,
                 config_path: str):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load BLAP configuration
        self.blap_config = BLAP2_Stage2_Config.from_file(config_path)
        
        # Initialize models
        self._init_clamp3(clamp3_weights_path)
        self._init_qformer_with_checkpoint(blap_checkpoint_path)
        self._init_t5()
        self._init_bottleneck_adapters()
        
        # Freeze all models except bottleneck adapters
        self._freeze_models()
        
        # Setup tokenizers
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(self.blap_config.LLM.t5_model)
        
        self.max_txt_len = self.blap_config.max_txt_len
        self.prompt = self.blap_config.prompt

        # Optimized loss weights: favor text generation over feature matching
        self.alpha = 0.3  # MSE loss weight (feature matching)
        self.beta = 1.0   # Cross-entropy loss weight (text generation)
        
    def _init_clamp3(self, weights_path: str):
        """Initialize and freeze CLaMP3 model"""
        print("🎵 Initializing CLaMP3...")
        
        # CLaMP3 configurations
        audio_config = BertConfig(
            vocab_size=1,
            hidden_size=AUDIO_HIDDEN_SIZE,
            num_hidden_layers=AUDIO_NUM_LAYERS,
            num_attention_heads=AUDIO_HIDDEN_SIZE//64,
            intermediate_size=AUDIO_HIDDEN_SIZE*4,
            max_position_embeddings=MAX_AUDIO_LENGTH
        )
        
        symbolic_config = BertConfig(
            vocab_size=1,
            hidden_size=M3_HIDDEN_SIZE,
            num_hidden_layers=PATCH_NUM_LAYERS,
            num_attention_heads=M3_HIDDEN_SIZE//64,
            intermediate_size=M3_HIDDEN_SIZE*4,
            max_position_embeddings=PATCH_LENGTH
        )
        
        # Initialize CLaMP3
        clamp3_model = CLaMP3Model(
            audio_config=audio_config,
            symbolic_config=symbolic_config,
            text_model_name=TEXT_MODEL_NAME,
            hidden_size=CLAMP3_HIDDEN_SIZE,
            load_m3=CLAMP3_LOAD_M3
        )
        
        # Load weights
        if weights_path and Path(weights_path).exists():
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint)
            
            # Handle module prefix
            if next(iter(state_dict)).startswith('module.'):
                state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            
            clamp3_model.load_state_dict(state_dict, strict=False)
            print("✅ CLaMP3 weights loaded successfully")
        
        self.clamp3 = clamp3_model
    
    def _init_qformer_with_checkpoint(self, checkpoint_path: str):
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
    
    def _init_t5(self):
        """Initialize and freeze T5 model"""
        print("📝 Initializing T5...")
        
        t5_config = T5Config.from_pretrained(self.blap_config.LLM.t5_model)
        t5_config.dense_act_fn = "gelu"
        
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            self.blap_config.LLM.t5_model, 
            config=t5_config
        )
        
        # T5 projection layer
        self.t5_proj = nn.Linear(
            self.qformer_config.hidden_size, 
            self.t5_model.config.hidden_size
        )
    
    def _init_bottleneck_adapters(self):
        """Initialize bottleneck adapters for Q-Former layers"""
        print("🔧 Initializing Bottleneck Adapters...")
        
        self.bottleneck_adapters = nn.ModuleDict()
        
        # Add adapters to all Q-Former layers (not just cross-attention)
        for layer_idx in range(self.qformer_config.num_hidden_layers):
            adapter_name = f"layer_{layer_idx}"
            
            # Improved bottleneck adapter: larger capacity for better learning
            adapter = BottleneckAdapter(
                clamp3_dim=768,                    # CLaMP3 *.npy feature dim
                qformer_dim=self.qformer_config.encoder_width,  # Q-Former encoder_width (1024)
                bottleneck_dim=128,                # Increased bottleneck dimension
                dropout=0.1
            )
            
            self.bottleneck_adapters[adapter_name] = adapter
            print(f"  Created bottleneck adapter for layer {layer_idx}: 768 -> 1024 -> 64 -> 1024")
        
        total_params = sum(p.numel() for p in self.bottleneck_adapters.parameters())
        print(f"Total bottleneck adapter parameters: {total_params:,}")
    
    def _freeze_models(self):
        """Freeze all models except bottleneck adapters"""
        print("🧊 Freezing models...")
        
        # Freeze CLaMP3
        for param in self.clamp3.parameters():
            param.requires_grad = False
        
        # Freeze Q-Former
        for param in self.qformer.parameters():
            param.requires_grad = False
        
        # Freeze query tokens
        self.query_tokens.requires_grad = False
        
        # Freeze T5
        for param in self.t5_model.parameters():
            param.requires_grad = False
        
        # Freeze T5 projection
        for param in self.t5_proj.parameters():
            param.requires_grad = False
        
        # Verify only bottleneck adapters are trainable
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def load_npy_features(self, audio_paths: List[str], max_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load *.npy features from file paths"""
        features_list = []
        masks_list = []
        
        for audio_path in audio_paths:
            # Load .npy file
            features = np.load(audio_path)
            
            # Convert to tensor
            features = torch.from_numpy(features).float()
            
            # Handle shape - ensure correct dimensions
            if features.ndim == 1:
                features = features.unsqueeze(0)  # Add sequence dimension
            elif features.ndim == 3:
                # If 3D, flatten to 2D (seq_len, feature_dim)
                features = features.view(-1, features.shape[-1])
            
            # Pad or truncate to max_length
            seq_len = features.shape[0]
            feature_dim = features.shape[1]
            
            if seq_len > max_length:
                features = features[:max_length]
                mask = torch.ones(max_length)
            elif seq_len < max_length:
                padding = torch.zeros(max_length - seq_len, feature_dim)
                features = torch.cat([features, padding], dim=0)
                mask = torch.zeros(max_length)
                mask[:seq_len] = 1
            else:
                mask = torch.ones(max_length)
            
            features_list.append(features)
            masks_list.append(mask)
        
        batch_features = torch.stack(features_list).to(self.device)
        batch_masks = torch.stack(masks_list).to(self.device)
        
        return batch_features, batch_masks
    
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Forward pass with proper adapter integration into Q-Former layers"""
        audio_paths = batch['audio_path']
        questions = batch['question']
        answers = batch['answer']
        
        batch_size = len(audio_paths)
        
        # 1. Load CLaMP3 *.npy features
        clamp3_features, audio_masks = self.load_npy_features(audio_paths)
        
        # 2. Initialize query tokens for Q-Former
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # 3. Apply bottleneck adapter for CLaMP3 → Q-Former feature adaptation
        adapter = self.bottleneck_adapters['layer_0']
        
        # Proper residual connection: use projected CLaMP3 as the "previous state"
        # This represents the baseline transformation that adapter will refine
        projected_clamp3 = adapter.projection(clamp3_features)  # (batch, seq, qformer_dim)
        
        # Adapter transformation: projected_clamp3 + adapter_refinement
        # adapter.forward internally does: residual + up = projected_clamp3 + adapter_output
        adapted_features = adapter(clamp3_features, projected_clamp3)
        
        # 4. Feature-level loss: adapter should improve upon simple projection
        # Encourage adapter to learn meaningful refinements
        feature_mse_loss = F.mse_loss(adapted_features, projected_clamp3.detach())
        
        # 4. Q-Former processing - allow gradients to flow through adapter output
        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=adapted_features,  # This needs gradients
            encoder_attention_mask=audio_masks,
            return_dict=True,
        )
        
        # Project to T5 space - keep gradients for adapter training
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long, device=self.device)
        
        # 5. T5 processing for end-to-end loss
        input_tokens = self.t5_tokenizer(
            [self.prompt.format(q) for q in questions],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        
        output_tokens = self.t5_tokenizer(
            answers,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        
        # T5 embeddings (frozen)
        with torch.no_grad():
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
        
        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )
        
        # T5 forward pass - need to allow gradients through inputs_t5
        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        
        # End-to-end cross-entropy loss (from T5)
        qa_loss = outputs.loss
        
        # 6. Dual loss combination (weighted)
        total_loss = self.alpha * feature_mse_loss + self.beta * qa_loss
        
        return {
            'loss': total_loss,
            'feature_mse_loss': feature_mse_loss,
            'qa_loss': qa_loss,
            'logits': outputs.logits,
            'adapted_features': adapted_features,
        }


def create_dataloaders(train_json: str, val_json: str, batch_size: int = 2) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for MusicQA dataset"""
    
    def collate_fn(batch):
        return {
            'audio_path': [item['audio_path'] for item in batch],
            'question': [item['question'] for item in batch],
            'answer': [item['answer'] for item in batch]
        }
    
    train_dataset = MusicQADataset(train_json, )
    val_dataset = MusicQADataset(val_json)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader


def train_bottleneck_adapters(
    model: SimpleBLAPAdapterTrainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    save_dir: str = "./bottleneck_adapter_checkpoints"
):
    """Train only the bottleneck adapters"""
    
    # Setup optimizer with improved parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),  # Default Adam betas
        eps=1e-8
    )
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.1
    )
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            outputs = model(batch)
            loss = outputs['loss']
            feature_loss = outputs['feature_mse_loss']
            qa_loss = outputs['qa_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({
                'total': f"{loss.item():.4f}",
                'feat': f"{feature_loss.item():.4f}",
                'qa': f"{qa_loss.item():.4f}"
            })
        
        avg_train_loss = np.mean(train_losses)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                outputs = model(batch)
                val_losses.append(outputs['loss'].item())
        
        avg_val_loss = np.mean(val_losses)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Save only bottleneck adapter weights
            bottleneck_state_dict = model.bottleneck_adapters.state_dict()
            
            torch.save({
                'epoch': epoch,
                'bottleneck_adapters': bottleneck_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_path / f'improved_adapter_mse_{model.alpha}_ce_{model.beta}_epochs_{args.epochs}_bs_{args.batch_size}_lr_{args.lr}.pth')
            
            print(f"✅ Saved best bottleneck adapters at epoch {epoch+1}")


def main():
    BASE = Path(__file__).resolve().parent.parent
    
    parser = argparse.ArgumentParser(description='Train BLAP Bottleneck Adapters')
    parser.add_argument('--blap_checkpoint', type=str,
        default=str(BASE / "blap" / "checkpoint" / "checkpoint.ckpt"),
        help='Path to BLAP checkpoint')
    parser.add_argument('--clamp3_weights', type=str,
        default=str(BASE / "clamp3" / "code" / "weights.pth"),  # 원하는 파일명/위치에 맞게 수정
        help='Path to CLaMP3 weights')
    parser.add_argument('--config_path', type=str,
        default=str(BASE / "blap" / "checkpoint" / "config.json"),
        help='Path to BLAP config')
    parser.add_argument('--train_json', type=str,
        default=str(BASE / "Adapters" / "PretrainMusicQA_npy.json"),
        help='Training data JSON')
    parser.add_argument('--val_json', type=str,
        default=str(BASE / "Adapters" / "EvalMusicQA_npy.json"),
        help='Validation data JSON')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (reduced for stability)')
    parser.add_argument('--epochs', type=int, default=12, help='Number of epochs (increased for better convergence)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (reduced for stability)')
    parser.add_argument('--save_dir', type=str,
        default=str(BASE / "projection_adapter_checkpoints"),
        help='Directory to save checkpoints')
    args = parser.parse_args()
    
    print("🚀 Starting Simple BLAP Projection Adapter Training")
    print("=" * 60)
    
    # Initialize model
    model = SimpleBLAPAdapterTrainer(
        blap_checkpoint_path=args.blap_checkpoint,
        clamp3_weights_path=args.clamp3_weights,
        config_path=args.config_path
    )
    
    model = model.to(model.device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.train_json, args.val_json, args.batch_size
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Start training
    train_bottleneck_adapters(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=args.save_dir
    )
    
    print("🎉 Training completed!")


if __name__ == "__main__":
    main()