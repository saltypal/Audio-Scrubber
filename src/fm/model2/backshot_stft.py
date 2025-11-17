"""
================================================================================
STFT-BASED CONV2D MODEL TRAINING SCRIPT
================================================================================

Training script for 2D U-Net Audio Denoiser with STFT spectrograms
- Loads clean and noisy audio pairs from dataset/instant
- Converts audio to STFT magnitude spectrograms
- Trains the model to predict clean spectrograms from noisy input
- Supports checkpoint resume capability

Usage:
    python src/model2/backshot_stft.py              # Start fresh training
    python src/model2/backshot_stft.py resume       # Resume from checkpoint
    python src/model2/backshot_stft.py --device cuda # Use GPU

Created by Satya with Copilot @ 16/11/25
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from stft_net import UNet2D
from config import Paths, AudioSettings, TrainingConfig


class STFTConfig:
    """STFT-specific configuration for spectrogram generation"""
    
    # STFT parameters
    N_FFT = 2048              # FFT window size (higher = better frequency resolution, lower temporal res)
    HOP_LENGTH = 512          # Number of samples between successive frames (~23ms at 22050 Hz)
    WIN_LENGTH = 2048         # Window length for STFT
    
    # This gives us:
    # - Frequency bins: n_fft // 2 + 1 = 1025
    # - Time resolution: hop_length / sample_rate ≈ 23ms
    # - Audio length: ~3.1 seconds for 256 time frames
    
    # Audio parameters
    SAMPLE_RATE = AudioSettings.SAMPLE_RATE  # 22050 Hz
    AUDIO_LENGTH = 44096  # ~2 seconds of audio (1024 samples shorter than 1D model)
    
    # Training parameters
    BATCH_SIZE = TrainingConfig.BATCH_SIZE
    NUM_EPOCHS = 10  # Reduced from 40 for faster testing
    LEARNING_RATE = TrainingConfig.LEARNING_RATE
    TRAIN_SPLIT = TrainingConfig.TRAIN_SPLIT
    
    # Device
    DEVICE = TrainingConfig.DEVICE


class STFTAudioDataset(Dataset):
    """
    Dataset for loading clean and noisy audio pairs and converting to STFT spectrograms.
    
    Key differences from 1D dataset:
    - Loads audio files
    - Converts to STFT magnitude spectrograms (no phase initially)
    - Each sample is (noisy_spectrogram, clean_spectrogram)
    - Both are tensors of shape (1, freq_bins, time_frames)
    """
    
    def __init__(self, clean_dir, noisy_dir, audio_length=44096, sample_rate=22050,
                 n_fft=2048, hop_length=512):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        print("Loading clean and noisy audio files...")
        
        # Get all clean files (recursively)
        clean_path = Path(clean_dir)
        self.clean_files = sorted([str(f.relative_to(clean_path)) for f in clean_path.rglob('*.flac')])
        
        # Get all noisy files (recursively)
        noisy_path = Path(noisy_dir)
        self.noisy_files = sorted([str(f.relative_to(noisy_path)) for f in noisy_path.rglob('*.flac')])
        
        print(f"  Found {len(self.clean_files)} clean files")
        print(f"  Found {len(self.noisy_files)} noisy files")
        print(f"  STFT params: n_fft={n_fft}, hop_length={hop_length}")
        print(f"  Frequency bins: {n_fft // 2 + 1}")
    
    def __len__(self):
        return len(self.noisy_files)
    
    def _audio_to_stft(self, audio_path):
        """Load audio and convert to STFT magnitude spectrogram"""
        import scipy.signal
        
        audio = None
        
        # Try multiple methods to load audio
        # Method 1: Try scipy's waveform reader
        try:
            import scipy.io.wavfile as wavfile
            sr_orig, audio_data = wavfile.read(audio_path)
            audio = audio_data.astype(np.float32)
            if len(audio.shape) > 1:  # stereo to mono
                audio = np.mean(audio, axis=1)
            # Resample if needed
            if sr_orig != self.sample_rate:
                num_samples = int(len(audio) * self.sample_rate / sr_orig)
                audio = scipy.signal.resample(audio, num_samples)
        except:
            pass
        
        # Method 2: Try librosa (if soundfile is working)
        if audio is None:
            try:
                audio, sr_loaded = librosa.load(audio_path, sr=self.sample_rate)
            except:
                pass
        
        # Method 3: Try audioread + scipy resample
        if audio is None:
            try:
                import audioread
                with audioread.audio_open(audio_path) as f:
                    sr_orig = f.samplerate
                    # Read all frames
                    frames = []
                    for buf in f:
                        frame_data = np.frombuffer(buf, np.int16).astype(np.float32)
                        frames.append(frame_data)
                    if frames:
                        audio = np.concatenate(frames)
                        # Normalize to [-1, 1]
                        audio = audio / 32768.0
                        if len(audio.shape) > 1:
                            audio = np.mean(audio, axis=1)
                        # Resample if needed
                        if sr_orig != self.sample_rate:
                            num_samples = int(len(audio) * self.sample_rate / sr_orig)
                            audio = scipy.signal.resample(audio, num_samples)
            except Exception as e:
                pass
        
        # If all else fails, return silence
        if audio is None:
            audio = np.zeros(self.audio_length, dtype=np.float32)
        
        # Ensure correct length (pad or truncate)
        if len(audio) < self.audio_length:
            audio = np.pad(audio, (0, self.audio_length - len(audio)), mode='constant')
        else:
            audio = audio[:self.audio_length]
        
        # Compute STFT using scipy: shape (freq_bins, time_frames)
        import scipy.signal
        freqs, times, stft_matrix = scipy.signal.spectrogram(
            audio, fs=self.sample_rate, nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length, scaling='spectrum'
        )
        
        # Get magnitude spectrogram
        magnitude = np.abs(stft_matrix)  # (freq_bins, time_frames)
        
        # Add channel dimension: (1, freq_bins, time_frames)
        magnitude = np.expand_dims(magnitude, axis=0)
        
        # Convert to float32 tensor
        return torch.FloatTensor(magnitude)
    
    def __getitem__(self, idx):
        """
        Load a noisy audio file, its clean version, and convert both to STFT spectrograms.
        """
        # Load noisy audio as STFT
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        noisy_stft = self._audio_to_stft(noisy_path)
        
        # Extract the clean filename from noisy filename
        noisy_filename = Path(self.noisy_files[idx]).name
        
        # Remove SNR suffix to get clean filename
        if '_snr_' in noisy_filename:
            clean_filename = noisy_filename.split('_snr_')[0] + '.flac'
        else:
            # Fallback for old format
            if '_noise_level_' in noisy_filename:
                clean_filename = noisy_filename.split('_noise_level_')[0] + '.flac'
            else:
                clean_filename = noisy_filename
        
        # Reconstruct clean file path (same directory structure)
        noisy_rel_path = Path(self.noisy_files[idx])
        clean_rel_path = noisy_rel_path.parent / clean_filename
        
        clean_path = os.path.join(self.clean_dir, str(clean_rel_path))
        
        # Load clean audio as STFT
        if os.path.exists(clean_path):
            clean_stft = self._audio_to_stft(clean_path)
        else:
            # Fallback: use noisy STFT if clean not found (shouldn't happen)
            clean_stft = noisy_stft
        
        return noisy_stft, clean_stft


def create_data_loaders(clean_dir, noisy_dir, batch_size=8, train_split=0.8):
    """Create train and validation data loaders"""
    
    # Create dataset
    dataset = STFTAudioDataset(
        clean_dir, noisy_dir,
        audio_length=STFTConfig.AUDIO_LENGTH,
        sample_rate=STFTConfig.SAMPLE_RATE,
        n_fft=STFTConfig.N_FFT,
        hop_length=STFTConfig.HOP_LENGTH
    )
    
    # Split into train and validation
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")
    
    for noisy_stft, clean_stft in pbar:
        noisy_stft = noisy_stft.to(device)
        clean_stft = clean_stft.to(device)
        
        # Forward pass
        predicted_clean = model(noisy_stft)
        
        # Compute loss
        loss = criterion(predicted_clean, clean_stft)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for noisy_stft, clean_stft in pbar:
            noisy_stft = noisy_stft.to(device)
            clean_stft = clean_stft.to(device)
            
            # Forward pass
            predicted_clean = model(noisy_stft)
            
            # Compute loss
            loss = criterion(predicted_clean, clean_stft)
            total_loss += loss.item()
            pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train_model(model_path=None, resume=False, device=None):
    """Main training loop"""
    
    # Device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = UNet2D(in_channels=1, out_channels=1).to(device)
    print(f"\n{'='*60}")
    print(f"STFT-based Conv2D U-Net Model")
    print(f"{'='*60}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=STFTConfig.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=TrainingConfig.SCHEDULER_PATIENCE, verbose=True
    )
    
    # Paths
    if model_path is None:
        model_path = str(Paths.MODEL_ROOT / "stft_unet2d_best.pth")
    
    checkpoint_path = str(Paths.MODEL_ROOT / "checkpoints" / "stft_latest_checkpoint.pth")
    
    # Create directories
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs_no_improve = 0
    
    # Resume from checkpoint if requested
    if resume and Path(checkpoint_path).exists():
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        epochs_no_improve = checkpoint['epochs_no_improve']
        
        print(f"  Resumed from epoch {start_epoch}")
        print(f"  Best val loss so far: {best_val_loss:.6f}")
    else:
        print(f"\nStarting fresh training")
    
    # Create data loaders
    print(f"\nPreparing dataset from: {Paths.INSTANT_CLEAN} and {Paths.INSTANT_NOISY}")
    train_loader, val_loader = create_data_loaders(
        str(Paths.INSTANT_CLEAN),
        str(Paths.INSTANT_NOISY),
        batch_size=STFTConfig.BATCH_SIZE,
        train_split=STFTConfig.TRAIN_SPLIT
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, STFTConfig.NUM_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, STFTConfig.NUM_EPOCHS)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Print stats
        print(f"\n[Epoch {epoch+1}/{STFTConfig.NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        
        # Step scheduler
        scheduler.step(val_loss)
        
        # Save checkpoint every epoch
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_no_improve': epochs_no_improve
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            model_data = {
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }
            torch.save(model_data, model_path)
            print(f"  ✅ Best model saved! (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{TrainingConfig.EARLY_STOPPING_PATIENCE} epochs")
            
            if epochs_no_improve >= TrainingConfig.EARLY_STOPPING_PATIENCE:
                print(f"\n⏹️ Early stopping triggered!")
                break
        
        print()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best model saved to: {model_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total epochs trained: {epoch + 1}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    return model_path


def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss curves"""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('STFT Conv2D Model - Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = str(Paths.OUTPUT_ROOT / "stft_training_history.png")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"✅ Training history saved to: {output_path}")
        plt.close()
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train STFT-based Conv2D Audio Denoiser")
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--model', type=str, help='Path to save model')
    
    args = parser.parse_args()
    
    # Handle resume from positional argument (for backward compatibility)
    import sys as sys_module
    if len(sys_module.argv) > 1 and sys_module.argv[1] == 'resume':
        args.resume = True
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    train_model(model_path=args.model, resume=args.resume, device=device)
