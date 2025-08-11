"""
BLAP LoRA Training Script with Loss Visualization
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import json
from typing import List, Dict

# Add paths
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/Adapters")

from blap_lora_trainer import BLAPLoRATrainer, MusicQADataset


def create_dataloaders(train_json: str, val_json: str, batch_size: int = 4):
    """Create training and validation dataloaders"""
    
    def collate_fn(batch):
        return {
            'audio_path': [item['audio_path'] for item in batch],
            'question': [item['question'] for item in batch],
            'answer': [item['answer'] for item in batch]
        }
    
    train_dataset = MusicQADataset(train_json)
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


def plot_loss_curves(train_losses: List[float], val_losses: List[float], 
                     learning_rates: List[float], epochs: List[int], save_path: Path):
    """Plot and save training curves"""
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('BLAP LoRA Fine-tuning: Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add loss reduction info
    if len(train_losses) > 1:
        train_reduction = ((train_losses[0] - train_losses[-1]) / train_losses[0]) * 100
        val_reduction = ((val_losses[0] - val_losses[-1]) / val_losses[0]) * 100
        
        info_text = f'Train Loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f} ({train_reduction:.1f}% ↓)\\n'
        info_text += f'Val Loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f} ({val_reduction:.1f}% ↓)'
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Learning rate schedule
    ax2.plot(epochs, learning_rates, 'g-', label='Learning Rate', linewidth=2, marker='^', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = save_path / 'training_curves.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training curves saved: {plot_file}")
    
    # Also save loss data as JSON
    loss_data = {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'train_loss_reduction_pct': train_reduction if len(train_losses) > 1 else 0,
        'val_loss_reduction_pct': val_reduction if len(val_losses) > 1 else 0
    }
    
    with open(save_path / 'loss_history.json', 'w') as f:
        json.dump(loss_data, f, indent=2)
    
    return plot_file


def save_lora_checkpoint(model: BLAPLoRATrainer, optimizer, scheduler, 
                        epoch: int, train_loss: float, val_loss: float, 
                        save_path: Path, is_best: bool = False):
    """Save LoRA checkpoint"""
    
    checkpoint_name = f'lora_epoch_{epoch+1}_val_{val_loss:.4f}.pt'
    if is_best:
        checkpoint_name = f'best_lora_epoch_{epoch+1}.pt'
    
    checkpoint_path = save_path / checkpoint_name
    
    # Save LoRA adapters separately
    lora_dir = save_path / f'lora_adapters_epoch_{epoch+1}'
    lora_dir.mkdir(exist_ok=True)
    
    # Save Q-Former LoRA
    model.qformer.save_pretrained(lora_dir / "qformer")
    
    # Save T5 LoRA
    model.t5_model.save_pretrained(lora_dir / "t5")
    
    # Save training state
    torch.save({
        'epoch': epoch,
        'model_state_dict': {
            'audio_projection': model.audio_projection.state_dict(),
            'query_tokens': model.query_tokens,
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'lora_config': {
            'r': model.lora_r,
            'alpha': model.lora_alpha,
            'dropout': model.lora_dropout
        }
    }, checkpoint_path)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def train_lora_model(
    model: BLAPLoRATrainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args
):
    """Main training function"""
    
    # Setup optimizer (only LoRA parameters + audio projection)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    verbose=True,
    min_lr=1e-6
    )
    
    # Create save directory
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Training tracking
    best_val_loss = float('inf')
    train_losses_history = []
    val_losses_history = []
    learning_rates_history = []
    epochs_list = []
    
    print("🚀 Starting LoRA Training")
    print("=" * 60)
    print(f"Device: {model.device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"LoRA config: r={model.lora_r}, alpha={model.lora_alpha}")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        # ============ TRAINING ============
        model.train()
        train_losses = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch_idx, batch in enumerate(train_pbar):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # Optimizer step
            optimizer.step()
            
            # Track loss
            train_losses.append(loss.item())
            current_loss = np.mean(train_losses[-10:])  # Moving average
            
            train_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_train_loss = np.mean(train_losses)
        
        # ============ VALIDATION ============
        model.eval()
        val_losses = []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_pbar:
                outputs = model(batch)
                loss = outputs['loss']
                val_losses.append(loss.item())
                
                val_pbar.set_postfix({
                    'val_loss': f'{np.mean(val_losses):.4f}'
                })
        
        avg_val_loss = np.mean(val_losses)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # ============ LOGGING ============
        print(f"\\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  LR:         {current_lr:.2e}")
        
        # Track metrics
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        learning_rates_history.append(current_lr)
        epochs_list.append(epoch + 1)
        
        # ============ SAVE CHECKPOINT ============
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            print(f"  🎉 New best validation loss!")
        
        # Save checkpoint
        save_lora_checkpoint(
            model, optimizer, scheduler,
            epoch, avg_train_loss, avg_val_loss,
            save_path, is_best=is_best
        )
        
        # ============ PLOT CURVES ============
        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            plot_loss_curves(
                train_losses_history, val_losses_history, 
                learning_rates_history, epochs_list, save_path
            )
    
    print("\\n🎉 Training completed!")
    print(f"📁 Results saved to: {save_path}")
    print(f"🏆 Best validation loss: {best_val_loss:.6f}")
    
    return train_losses_history, val_losses_history


def main():
    BASE = Path(__file__).resolve().parent.parent
    
    parser = argparse.ArgumentParser(description='Train BLAP with LoRA')
    
    # Model paths
    parser.add_argument('--blap_checkpoint', type=str,
        default=str(BASE / "blap" / "checkpoint" / "checkpoint.ckpt"),
        help='Path to BLAP checkpoint')
    parser.add_argument('--config_path', type=str,
        default=str(BASE / "blap" / "checkpoint" / "config.json"),
        help='Path to BLAP config')
    parser.add_argument("--t5_model", type=str, default="google/flan-t5-base")

    # Data paths
    parser.add_argument('--train_json', type=str,
        default=str(BASE / "Adapters" / "FinetuneMusicQA_npy.json"),
        help='Training data JSON')
    parser.add_argument('--val_json', type=str,
        default=str(BASE / "Adapters" / "EvalMusicQA_npy.json"),
        help='Validation data JSON')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, 
        help='Batch size (adjust based on GPU memory)')
    parser.add_argument('--epochs', type=int, default=20, 
        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, 
        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
        help='Weight decay')
    
    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=16, 
        help='LoRA rank (lower = fewer parameters)')
    parser.add_argument('--lora_alpha', type=int, default=32, 
        help='LoRA alpha (scaling factor)')
    parser.add_argument('--lora_dropout', type=float, default=0.1, 
        help='LoRA dropout')
    
    # Output
    parser.add_argument('--save_dir', type=str,
        default=str(BASE / "lora_results"),
        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Initialize model
    print("🤖 Initializing BLAP LoRA model...")
    model = BLAPLoRATrainer(
        blap_checkpoint_path=args.blap_checkpoint,
        config_path=args.config_path,
        t5_model_name=args.t5_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    model = model.to(model.device)
    
    # Create dataloaders
    print("📊 Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        args.train_json, args.val_json, args.batch_size
    )
    
    # Start training
    train_losses, val_losses = train_lora_model(
        model, train_loader, val_loader, args
    )
    
    print("\\n✅ All done! Check your results in:", args.save_dir)


if __name__ == "__main__":
    main()