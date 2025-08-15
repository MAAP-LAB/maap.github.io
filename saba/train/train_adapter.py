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
from torch.cuda.amp import GradScaler, autocast

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/clamp3/code")
sys.path.append(PROJECT_ROOT + "/blap")

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


class AdapterTrainer:
    """Trainer for SABA Adapter with frozen QFormer"""
    
    def __init__(self, 
                 blap_checkpoint_path: str,
                 bottleneck_dim: int = 128,
                 learning_rate: float = 1e-4,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.bottleneck_dim = bottleneck_dim
        self.learning_rate = learning_rate
        
        # Initialize SABA QFormer with frozen weights
        from saba.SABA_qformer import SABA_QFormer
        
        self.model = SABA_QFormer(
            blap_checkpoint_path=blap_checkpoint_path,
            bottleneck_dim=bottleneck_dim,
            device=device
        )
        
        # Setup optimizer for adapter only
        adapter_params = list(self.model.get_adapter_parameters())
        projection_params = list(self.model.audio_proj.parameters()) + \
                          list(self.model.text_proj.parameters()) + \
                          list(self.model.symbolic_proj.parameters())
        
        trainable_params = adapter_params + projection_params
        

        # Mixed precision training
        self.scaler = GradScaler()

        for p in self.parameters():
            p.requires_grad = False

        # 2) 학습시킬 모듈만 해제
        for mod in [self.model.adapter,
            self.model.audio_proj, self.model.text_proj, self.model.symbolic_proj,
            self.model.atm_head]:
            for p in mod.parameters():
                p.requires_grad = True

        if hasattr(self.model, "logit_scale"):
            self.model.logit_scale.requires_grad = True
        elif hasattr(self.model, "temperature"):
            self.model.temperature.requires_grad = True

        self.query_tokens.requires_grad = False

        self.optimizer = self.build_adapter_proj_optimizer(lr=2e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=True
        )

        # 디버그: 학습 가능한 파라미터 확인
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"🎯 Trainable params: {trainable:,} / {total:,}")
        
        print(f"🎯 AdapterTrainer initialized:")
        print(f"   Device: {device}")
        print(f"   Bottleneck dim: {bottleneck_dim}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Trainable adapter params: {self.model.get_trainable_adapter_parameters():,}")
        
    def load_audio_features(self, audio_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load CLaMP3 features from .npy files"""
        batch_features = []
        batch_masks = []
        
        for audio_path in audio_paths:
            if Path(audio_path).exists():
                # Load CLaMP3 features
                features = np.load(audio_path)  # Shape: [seq_len, feat_dim]
                features = torch.from_numpy(features).float()
                
                # Create attention mask (all valid)
                mask = torch.ones(features.shape[0], dtype=torch.long)
            else:
                # Fallback: dummy features if file missing
                print(f"⚠️ Missing audio file: {audio_path}")
                features = torch.randn(32, 256)  # Default CLaMP3 dims
                mask = torch.ones(32, dtype=torch.long)
            
            batch_features.append(features)
            batch_masks.append(mask)
        
        # Pad to max sequence length in batch
        max_seq_len = max(f.shape[0] for f in batch_features)
        feat_dim = batch_features[0].shape[1]
        
        padded_features = torch.zeros(len(batch_features), max_seq_len, feat_dim)
        padded_masks = torch.zeros(len(batch_features), max_seq_len)
        
        for i, (feat, mask) in enumerate(zip(batch_features, batch_masks)):
            seq_len = feat.shape[0]
            padded_features[i, :seq_len] = feat
            padded_masks[i, :seq_len] = mask
            
        return padded_features.to(self.device), padded_masks.to(self.device)
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step with adapter only"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Load audio features from .npy files
        audio_features, audio_masks = self.load_audio_features(batch['audio_path'])
        
        # Use answers as text for training
        texts = batch['answer']
        
        # Forward pass through SABA QFormer
        outputs = self.model(
            texts=texts,
            audio_embeds=audio_features,
            audio_atts=audio_masks,
            modality="audio"
        )
        
        # Extract losses
        total_loss = outputs.loss
        loss_atc = outputs.loss_atc or 0.0
        loss_atm = outputs.loss_atm or 0.0
        loss_lm = outputs.loss_lm or 0.0
        
        # Backward pass with mixed precision
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Return metrics
        return {
            'total_loss': total_loss.item(),
            'contrastive_loss': loss_atc.item() if torch.is_tensor(loss_atc) else loss_atc,
            'matching_loss': loss_atm.item() if torch.is_tensor(loss_atm) else loss_atm,
            'captioning_loss': loss_lm.item() if torch.is_tensor(loss_lm) else loss_lm,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        
        total_metrics = {
            'total_loss': 0.0,
            'contrastive_loss': 0.0,
            'matching_loss': 0.0,
            'captioning_loss': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Load audio features
                audio_features, audio_masks = self.load_audio_features(batch['audio_path'])
                texts = batch['answer']
                
                with autocast():
                    outputs = self.model(
                        texts=texts,
                        audio_embeds=audio_features,
                        audio_atts=audio_masks,
                        modality="audio"
                    )
                    
                    # Accumulate losses
                    total_metrics['total_loss'] += outputs.loss.item()
                    total_metrics['contrastive_loss'] += (outputs.loss_atc.item() if outputs.loss_atc else 0.0)
                    total_metrics['matching_loss'] += (outputs.loss_atm.item() if outputs.loss_atm else 0.0)
                    total_metrics['captioning_loss'] += (outputs.loss_lm.item() if outputs.loss_lm else 0.0)
                    
                num_batches += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= max(num_batches, 1)
        
        return total_metrics
    
    def build_adapter_proj_optimizer(self, lr: float = 2e-4, weight_decay: float = 0.01):
        decay_params, no_decay_params = [], []
        temp_param = []

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # logit_scale/temperature는 별 그룹(무 decay)
            if hasattr(self.model, "logit_scale") and p is self.model.logit_scale:
                temp_param.append(p); continue
            if hasattr(self.model, "temperature") and p is self.model.temperature:
                temp_param.append(p); continue

            # no-decay: bias, norm류(1D), LN/BN 등
            if p.dim() == 1 or n.endswith(".bias") or "norm" in n.lower() or "ln" in n.lower() or "bn" in n.lower():
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        if temp_param:
            param_groups.append({"params": temp_param, "lr": lr, "weight_decay": 0.0})

        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
        return optimizer

    
    def save_adapter(self, save_path: str):
        """Save only the adapter weights"""
        adapter_state = {
            'adapter_state_dict': self.model.adapter.state_dict(),
            'bottleneck_dim': self.bottleneck_dim,
            'learning_rate': self.learning_rate,
            'projection_layers': {
                'audio_proj': self.model.audio_proj.state_dict(),
                'text_proj': self.model.text_proj.state_dict(),
                'symbolic_proj': self.model.symbolic_proj.state_dict()
            }
        }
        
        torch.save(adapter_state, save_path)
        print(f"💾 Adapter saved to: {save_path}")


def train_adapter_with_musicqa(
    train_json: str,
    val_json: str,
    blap_checkpoint: str,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    bottleneck_dim: int = 128,
    save_dir: str = "adapter_checkpoints"
):
    """
    Main training function for SABA adapter using MusicQA dataset
    """
    # Setup logging (once)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_json, val_json, batch_size)
    logger.info(f"📊 Dataloaders ready — train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # Initialize trainer
    trainer = AdapterTrainer(
        blap_checkpoint_path=blap_checkpoint,
        bottleneck_dim=bottleneck_dim,
        learning_rate=learning_rate,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    best_val_loss = float("inf")
    best_model_path = None

    try:
        for epoch in range(num_epochs):
            logger.info(f"\n🔥 Epoch {epoch + 1}/{num_epochs}")

            # ---- Training ----
            trainer.model.train()
            epoch_metrics = {
                "total_loss": 0.0,
                "contrastive_loss": 0.0,
                "matching_loss": 0.0,
                "captioning_loss": 0.0,
            }

            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", dynamic_ncols=True)
            for batch_idx, batch in enumerate(train_pbar):
                metrics = trainer.train_step(batch)

                # Accumulate metrics
                for k in epoch_metrics:
                    epoch_metrics[k] += metrics[k]

                train_pbar.set_postfix({
                    "loss": f"{metrics['total_loss']:.4f}",
                    "lr": f"{metrics['learning_rate']:.2e}",
                })

            # Average training metrics
            num_train_batches = max(len(train_loader), 1)
            for k in epoch_metrics:
                epoch_metrics[k] /= num_train_batches

            logger.info(
                "📈 Train — "
                f"Loss: {epoch_metrics['total_loss']:.4f} | "
                f"ATC: {epoch_metrics['contrastive_loss']:.4f} | "
                f"ATM: {epoch_metrics['matching_loss']:.4f} | "
                f"LM: {epoch_metrics['captioning_loss']:.4f}"
            )

            # ---- Validation ----
            val_metrics = trainer.validate(val_loader)
            logger.info(
                "📉 Val   — "
                f"Loss: {val_metrics['total_loss']:.4f} | "
                f"ATC: {val_metrics['contrastive_loss']:.4f} | "
                f"ATM: {val_metrics['matching_loss']:.4f} | "
                f"LM: {val_metrics['captioning_loss']:.4f}"
            )

            # Step LR scheduler (ReduceLROnPlateau expects a metric)
            if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
                trainer.scheduler.step(val_metrics["total_loss"])
                current_lr = trainer.optimizer.param_groups[0]["lr"]
                logger.info(f"🪜 LR after scheduler step: {current_lr:.2e}")

            # ---- Checkpointing ----
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                best_model_path = save_path / f"best_adapter_epoch_{epoch + 1}.pth"
                trainer.save_adapter(str(best_model_path))
                logger.info("🌟 New best model saved!")

            # (Optional) periodic checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = save_path / f"adapter_checkpoint_epoch_{epoch + 1}.pth"
                trainer.save_adapter(str(checkpoint_path))
                logger.info(f"💾 Periodic checkpoint saved: {checkpoint_path.name}")

        logger.info(f"🎉 Training completed! Best val loss: {best_val_loss:.4f}")
        if best_model_path is not None:
            logger.info(f"🏆 Best checkpoint: {best_model_path}")

    except KeyboardInterrupt:
        logger.warning("⛔ Training interrupted by user.")
    except Exception as e:
        logger.exception(f"❌ Error during training: {e}")

    return trainer



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SABA Adapter with MusicQA")
    parser.add_argument("--train_json", required=True, help="Path to training JSON")
    parser.add_argument("--val_json", required=True, help="Path to validation JSON") 
    parser.add_argument("--blap_checkpoint", required=True, help="Path to QFormer checkpoint")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--bottleneck_dim", type=int, default=128, help="Adapter bottleneck dimension")
    parser.add_argument("--save_dir", default="adapter_checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    train_adapter_with_musicqa(
        train_json=args.train_json,
        val_json=args.val_json,
        blap_checkpoint=args.blap_checkpoint,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        bottleneck_dim=args.bottleneck_dim,
        save_dir=args.save_dir
    )