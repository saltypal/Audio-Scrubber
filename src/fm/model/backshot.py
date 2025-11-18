import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import numpy as np
from pathlib import Path
# Import UNet1D - handle both direct execution and module import
try:
    from .neuralnet import UNet1D
except ImportError:
    # When run directly, add src to path
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(src_path))
    from fm.model.neuralnet import UNet1D
from tqdm import tqdm
import matplotlib.pyplot as plt
import soundfile as sf

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import Paths, AudioSettings, TrainingConfig

"""
Memory-Optimized Training Script for 1D U-Net Audio Denoiser
- Minimal RAM usage with streaming data loading
- CUDA support with automatic device detection
- On-the-fly noise generation with real FM noise
- Simple and efficient
"""

# --- Configuration ---
class Config:
    # Paths
    CLEAN_AUDIO_DIR = str(Paths.LIBRISPEECH_DEV_CLEAN)
    NOISE_FILE = str(Paths.NOISE_PURE)
    MODEL_SAVE_PATH = str(Paths.MODEL_FM_BEST)
    CHECKPOINT_DIR = str(Paths.MODEL_FM_CHECKPOINTS)
    
    # Audio parameters
    SAMPLE_RATE = AudioSettings.SAMPLE_RATE
    AUDIO_LENGTH = AudioSettings.AUDIO_LENGTH
    
    # Training hyperparameters - MEMORY OPTIMIZED
    BATCH_SIZE = 2  # Small batch for low RAM usage
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    TRAIN_SPLIT = 0.9
    
    # Model parameters
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    
    # Device - Auto-detect CUDA
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MemoryEfficientDataset(Dataset):
    """
    Ultra-lightweight dataset with minimal memory footprint.
    - No pre-loading of audio files
    - Noise loaded once and shared
    - Streaming data loading
    """
    
    def __init__(self, clean_dir, noise_file, audio_length=44096, sample_rate=22050, 
                 snr_db_range=(5, 20)):
        self.clean_dir = clean_dir
        self.noise_file = noise_file
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.snr_db_range = snr_db_range
        self.noise_audio = None  # Lazy load
        
        # Get all clean files
        clean_path = Path(clean_dir)
        self.clean_files = sorted([str(f) for f in clean_path.rglob('*.flac')])
        
        print(f"[Dataset] Found {len(self.clean_files)} clean files")
        print(f"[Dataset] Audio length: {audio_length} samples ({audio_length/sample_rate:.2f}s)")
        print(f"[Dataset] SNR range: {snr_db_range[0]}-{snr_db_range[1]} dB")
    
    def _load_noise_once(self):
        """Load noise file once and cache it"""
        if self.noise_audio is not None:
            return
        
        print(f"[Dataset] Loading noise file...")
        
        # Load only enough noise samples (not the entire file)
        max_samples = self.audio_length * 20  # 20 segments
        self.noise_audio, _ = sf.read(self.noise_file, stop=max_samples, dtype='float32')
        
        # Convert to mono if needed
        if len(self.noise_audio.shape) > 1:
            self.noise_audio = np.mean(self.noise_audio, axis=1)
        
        # Tile if too short
        if len(self.noise_audio) < self.audio_length * 5:
            repeats = (self.audio_length * 5) // len(self.noise_audio) + 1
            self.noise_audio = np.tile(self.noise_audio, repeats)[:max_samples]
        
        print(f"[Dataset] Noise loaded: {len(self.noise_audio)} samples, {self.noise_audio.nbytes/1024/1024:.1f} MB")
    
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        # Lazy load noise on first access
        if self.noise_audio is None:
            self._load_noise_once()
        
        # Load clean audio (stream, don't cache)
        clean_audio, sr = sf.read(self.clean_files[idx], dtype='float32')
        
        # Simple resampling if needed (most LibriSpeech is already 16kHz)
        if sr != self.sample_rate and sr == 16000 and self.sample_rate == 22050:
            # Quick nearest-neighbor resample for speed
            indices = np.linspace(0, len(clean_audio)-1, int(len(clean_audio) * self.sample_rate / sr))
            clean_audio = clean_audio[indices.astype(int)]
        
        # Crop or pad to target length
        if len(clean_audio) > self.audio_length:
            start = np.random.randint(0, len(clean_audio) - self.audio_length)
            clean_audio = clean_audio[start:start + self.audio_length]
        else:
            clean_audio = np.pad(clean_audio, (0, self.audio_length - len(clean_audio)))
        
        # Add noise with random SNR
        snr_db = np.random.uniform(self.snr_db_range[0], self.snr_db_range[1])
        noisy_audio = self._add_noise_snr(clean_audio, snr_db)
        
        # Convert to tensors
        return (
            torch.from_numpy(noisy_audio).unsqueeze(0),  # (1, L)
            torch.from_numpy(clean_audio).unsqueeze(0)   # (1, L)
        )
    
    def _add_noise_snr(self, signal, snr_db):
        """Add noise at specified SNR"""
        signal_power = np.mean(signal ** 2)
        
        # Get random noise segment
        if len(self.noise_audio) > len(signal):
            start = np.random.randint(0, len(self.noise_audio) - len(signal))
            noise = self.noise_audio[start:start + len(signal)]
        else:
            noise = self.noise_audio[:len(signal)]
        
        noise_power = np.mean(noise ** 2)
        
        # Calculate noise scaling
        if noise_power > 0:
            snr_linear = 10 ** (snr_db / 10.0)
            noise_scale = np.sqrt(signal_power / (snr_linear * noise_power))
        else:
            noise_scale = 0
        
        return signal + noise * noise_scale


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
    print(f"Training 1D U-Net Audio Denoiser")
    print(f"{'='*60}")
    print(f"Device: {config.DEVICE}")
    if config.DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"{'='*60}\n")
    
    # Create dataset
    dataset = MemoryEfficientDataset(
        config.CLEAN_AUDIO_DIR,
        config.NOISE_FILE,
        config.AUDIO_LENGTH,
        config.SAMPLE_RATE
    )
    
    # Split dataset
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"\nDataset split: {train_size} train, {val_size} val\n")
    
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
    model = UNet1D(in_channels=1, out_channels=1).to(config.DEVICE)
    
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
        if (epoch + 1) % 5 == 0:
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
    
    parser = argparse.ArgumentParser(description='Train 1D U-Net Audio Denoiser')
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
