"""
================================================================================
OFDM DENOISING MODEL TRAINER
================================================================================
Memory-optimized training for OFDM_UNet on IQ dataset from dataset/OFDM/

Features:
- Streaming data loading (minimal RAM usage)
- CUDA support with automatic device detection
- On-the-fly chunking of IQ samples
- Best model saving with checkpointing
- Training history visualization

Dataset Format:
- Files are raw complex64 IQ samples
- Shape: (N,) where N = total samples
- Converted to (2, 1024) tensors [I, Q channels]

Usage:
    python src/ofdm/model/train_ofdm.py
    python src/ofdm/model/train_ofdm.py --epochs 50 --batch-size 32
    python src/ofdm/model/train_ofdm.py --resume saved_models/OFDM/checkpoint_epoch_10.pth
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ofdm.model.neuralnet import OFDM_UNet


# --- Configuration ---
class Config:
    # Paths
    CLEAN_DATA_PATH = "dataset/OFDM/clean_ofdm.iq"
    NOISY_DATA_PATH = "dataset/OFDM/noisy_ofdm.iq"
    MODEL_SAVE_PATH = "saved_models/OFDM/unet1d_best.pth"
    CHECKPOINT_DIR = "saved_models/OFDM"
    
    # Training hyperparameters - MEMORY OPTIMIZED
    BATCH_SIZE = 4  # Small batch for low RAM usage
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.9
    
    # Model parameters
    IN_CHANNELS = 2  # I, Q
    OUT_CHANNELS = 2
    CHUNK_SIZE = 1024
    
    # Device - Auto-detect CUDA
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OFDMDataset(Dataset):
    """
    Memory-efficient IQ dataset with streaming load.
    - Loads IQ files once at init
    - Chunks on-the-fly during iteration
    - Minimal memory footprint
    """
    def __init__(self, clean_path, noisy_path, chunk_size=1024):
        """
        Args:
            clean_path: Path to clean_ofdm.iq file
            noisy_path: Path to noisy_ofdm.iq file
            chunk_size: Number of samples per training chunk
        """
        self.chunk_size = chunk_size
        
        print(f"[Dataset] Loading IQ files...")
        print(f"   Clean: {clean_path}")
        print(f"   Noisy: {noisy_path}")
        
        # Load IQ files (complex64 format)
        self.clean_data = np.fromfile(clean_path, dtype=np.complex64)
        self.noisy_data = np.fromfile(noisy_path, dtype=np.complex64)
        
        # Ensure both files have same length
        min_len = min(len(self.clean_data), len(self.noisy_data))
        self.clean_data = self.clean_data[:min_len]
        self.noisy_data = self.noisy_data[:min_len]
        
        # Calculate number of chunks
        self.num_chunks = min_len // chunk_size
        
        print(f"[Dataset] Loaded {min_len:,} samples")
        print(f"   Total chunks: {self.num_chunks:,}")
        print(f"   Chunk size: {chunk_size}")
        print(f"   Memory: {(self.clean_data.nbytes + self.noisy_data.nbytes) / 1024 / 1024:.1f} MB")
    
    def __len__(self):
        return self.num_chunks
    
    def __getitem__(self, idx):
        """Returns (noisy_tensor, clean_tensor) pair."""
        start = idx * self.chunk_size
        end = start + self.chunk_size
        
        clean_chunk = self.clean_data[start:end]
        noisy_chunk = self.noisy_data[start:end]
        
        # Convert to [2, chunk_size] format: [I, Q]
        clean_tensor = torch.stack([
            torch.from_numpy(clean_chunk.real),
            torch.from_numpy(clean_chunk.imag)
        ]).float()
        
        noisy_tensor = torch.stack([
            torch.from_numpy(noisy_chunk.real),
            torch.from_numpy(noisy_chunk.imag)
        ]).float()
        
        return noisy_tensor, clean_tensor

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for noisy, clean in tqdm(loader, desc="Training"):
        noisy, clean = noisy.to(device), clean.to(device)
        
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Free memory
        del noisy, clean, output, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for noisy, clean in tqdm(loader, desc="Validation"):
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)
            total_loss += loss.item()
            
            # Free memory
            del noisy, clean, output, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return total_loss / len(loader)

def train_model(config, resume_checkpoint=None):
    """Main training function"""
    
    print(f"\n{'='*60}")
    print(f"Training OFDM_UNet IQ Denoiser")
    print(f"{'='*60}")
    print(f"Device: {config.DEVICE}")
    if config.DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Chunk size: {config.CHUNK_SIZE}")
    print(f"{'='*60}\n")
    
    # Create dataset
    dataset = OFDMDataset(
        config.CLEAN_DATA_PATH,
        config.NOISY_DATA_PATH,
        config.CHUNK_SIZE
    )
    
    # Split dataset
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"\nDataset split: {train_size:,} train, {val_size:,} val\n")
    
    # Create dataloaders - MEMORY OPTIMIZED
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False  # Disable for low RAM
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Create model
    model = OFDM_UNet(in_channels=config.IN_CHANNELS, out_channels=config.OUT_CHANNELS).to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}\n")
    
    # Training loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        val_losses.append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr:.8f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(config.MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, config.MODEL_SAVE_PATH)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(config.CHECKPOINT_DIR) / f'checkpoint_epoch_{epoch+1}.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path.name}")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"{'='*60}\n")
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plot_path = Path(config.MODEL_SAVE_PATH).parent / 'training_history.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training plot saved to: {plot_path}")
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train OFDM_UNet IQ Denoiser')
    parser.add_argument('--epochs', '-e', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=None, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--resume', '-r', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--cpu', action='store_true', help='Force CPU (disable CUDA)')
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Apply command-line overrides
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
        print(f"Epochs set to: {args.epochs}")
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
        print(f"Batch size set to: {args.batch_size}")
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
        print(f"Learning rate set to: {args.learning_rate}")
    if args.cpu:
        config.DEVICE = torch.device('cpu')
        print("Forcing CPU mode")
    
    # Train
    train_model(config, resume_checkpoint=args.resume)

