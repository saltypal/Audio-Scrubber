"""
Optimized OFDM Training Loop with Mixed Precision and Smart Batching
Trains 1D U-Net for OFDM waveform denoising
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ofdm.model.neuralnet import OFDM_UNet


class OFDMStreamDataset(Dataset):
    """Streaming dataset for large IQ files"""
    
    def __init__(self, clean_path, noisy_path, chunk_size=1024, max_samples=None):
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.chunk_size = chunk_size
        
        # Get file lengths
        clean_data = np.fromfile(clean_path, dtype=np.complex64)
        noisy_data = np.fromfile(noisy_path, dtype=np.complex64)
        
        assert len(clean_data) == len(noisy_data), "Clean and noisy files must have same length"
        
        total_samples = len(clean_data)
        self.num_chunks = total_samples // chunk_size
        
        if max_samples:
            self.num_chunks = min(self.num_chunks, max_samples // chunk_size)
        
        print(f"Dataset: {self.num_chunks:,} chunks of {chunk_size} samples")
        print(f"Total: {self.num_chunks * chunk_size:,} samples ({self.num_chunks * chunk_size * 8 / 1024**2:.1f} MB)")
        
        # Preload for faster training (if fits in RAM)
        if total_samples * 8 < 4 * 1024**3:  # Less than 4GB
            print("Preloading dataset into RAM...")
            self.clean_data = clean_data
            self.noisy_data = noisy_data
            self.preloaded = True
        else:
            print("Using streaming mode (dataset too large for RAM)")
            self.preloaded = False
    
    def __len__(self):
        return self.num_chunks
    
    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = start + self.chunk_size
        
        if self.preloaded:
            clean = self.clean_data[start:end]
            noisy = self.noisy_data[start:end]
        else:
            # Stream from disk
            clean = np.fromfile(self.clean_path, dtype=np.complex64, 
                              count=self.chunk_size, offset=start*8)
            noisy = np.fromfile(self.noisy_path, dtype=np.complex64, 
                              count=self.chunk_size, offset=start*8)
        
        # Convert to [2, Length] format (I, Q channels)
        clean_tensor = torch.stack([
            torch.from_numpy(np.real(clean).astype(np.float32)),
            torch.from_numpy(np.imag(clean).astype(np.float32))
        ])
        
        noisy_tensor = torch.stack([
            torch.from_numpy(np.real(noisy).astype(np.float32)),
            torch.from_numpy(np.imag(noisy).astype(np.float32))
        ])
        
        return noisy_tensor, clean_tensor


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, use_amp=True):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for noisy, clean in pbar:
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler:
            with autocast():
                output = model(noisy)
                loss = criterion(output, clean)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc="Validating"):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            output = model(noisy)
            loss = criterion(output, clean)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_model(
    clean_path,
    noisy_path,
    epochs=100,
    batch_size=32,
    chunk_size=1024,
    lr=0.001,
    val_split=0.1,
    save_dir='saved_models/OFDM',
    use_amp=True,
    early_stopping_patience=20
):
    """Main training loop"""
    
    print("="*80)
    print("OFDM 1D U-NET TRAINING - OPTIMIZED")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    print(f"   Clean: {clean_path}")
    print(f"   Noisy: {noisy_path}")
    
    full_dataset = OFDMStreamDataset(clean_path, noisy_path, chunk_size=chunk_size)
    
    # Split train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training: {train_size:,} chunks")
    print(f"   Validation: {val_size:,} chunks")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print(f"\n🧠 Initializing model...")
    model = OFDM_UNet(in_channels=2, out_channels=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Chunk size: {chunk_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Mixed precision: {use_amp and device.type == 'cuda'}")
    print(f"   Early stopping patience: {early_stopping_patience}")
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print("="*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-"*80)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f"\nResults:")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss:   {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'chunk_size': chunk_size
            }
            
            best_path = save_dir / 'ofdm_unet_best.pth'
            torch.save(checkpoint, best_path)
            print(f"   ✅ Saved best model: {best_path}")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{early_stopping_patience})")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {save_dir / 'ofdm_unet_best.pth'}")
    print("\n🎓 Ready for inference!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Train OFDM denoising model')
    
    parser.add_argument('--clean-data', type=str, required=True,
                       help='Path to clean IQ file')
    parser.add_argument('--noisy-data', type=str, required=True,
                       help='Path to noisy IQ file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--chunk-size', type=int, default=1024,
                       help='Chunk size in samples (default: 1024)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split (default: 0.1)')
    parser.add_argument('--save-dir', type=str, default='saved_models/OFDM',
                       help='Save directory (default: saved_models/OFDM)')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience (default: 20)')
    
    args = parser.parse_args()
    
    train_model(
        clean_path=args.clean_data,
        noisy_path=args.noisy_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        lr=args.lr,
        val_split=args.val_split,
        save_dir=args.save_dir,
        use_amp=not args.no_amp,
        early_stopping_patience=args.patience
    )


if __name__ == "__main__":
    main()
