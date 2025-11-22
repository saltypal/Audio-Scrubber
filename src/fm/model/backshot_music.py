"""
Music Audio Denoiser Training Script (backshot_music.py)

Specialized training script for music denoising using 1D U-Net.
Optimized for music characteristics: full frequency spectrum, dense audio, polyphonic.

This version mirrors the speech/FM pipeline and uses **on-the-fly
noise mixing** with the real FM noise file (e.g. superNoiseFM.wav),
instead of requiring pre-generated noisy music files.

Key Differences from Speech Training:
- Higher sample rate: 44100 Hz (CD quality)
- Longer audio segments: 88192 samples (~2 seconds)
- Music-specific dataset paths
- Same architecture (U-Net works for both)

Usage:
    # Train music model with on-the-fly noise
    python src/fm/model/backshot_music.py
    
    # Resume training
    python src/fm/model/backshot_music.py resume

Created by Satya with Copilot @ 15/11/25,
updated for on-the-fly FM noise mixing.
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
from neuralnet import UNet1D
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root (where config_music.py lives) to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from config_music import Paths, AudioSettings, TrainingConfig, NoiseSettings


class MusicConfig:
    """Configuration for music model training"""

    # Paths - Using raw music as clean, FM noise as background
    CLEAN_AUDIO_DIRS = [str(Paths.MUSIC_ROOT)]
    NOISE_FILE = str(Paths.NOISE_ROOT / "superNoiseFM.wav")
    MODEL_SAVE_PATH = "saved_models/music_100_norm/unet1d_best_music.pth"

    # Audio parameters - Music specific (44.1kHz)
    SAMPLE_RATE = AudioSettings.SAMPLE_RATE  # 44100 Hz
    AUDIO_LENGTH = AudioSettings.AUDIO_LENGTH  # 88192 samples

    # Training hyperparameters
    BATCH_SIZE = TrainingConfig.BATCH_SIZE
    NUM_EPOCHS = 100  # Full training
    LEARNING_RATE = TrainingConfig.LEARNING_RATE
    TRAIN_SPLIT = TrainingConfig.TRAIN_SPLIT

    # Optimizer settings
    OPTIMIZER = TrainingConfig.OPTIMIZER
    SCHEDULER_PATIENCE = TrainingConfig.SCHEDULER_PATIENCE
    SCHEDULER_FACTOR = TrainingConfig.SCHEDULER_FACTOR
    EARLY_STOPPING_PATIENCE = TrainingConfig.EARLY_STOPPING_PATIENCE

    # Checkpoint settings
    SAVE_CHECKPOINT_EVERY = TrainingConfig.SAVE_CHECKPOINT_EVERY
    CHECKPOINT_DIR = TrainingConfig.CHECKPOINT_DIR

    # Model parameters
    IN_CHANNELS = AudioSettings.IN_CHANNELS
    OUT_CHANNELS = AudioSettings.OUT_CHANNELS

    # Device
    DEVICE = TrainingConfig.DEVICE


class OnFlyMusicNoiseDataset(Dataset):
    """On-the-fly noisy music generator (clean music + FM noise).

    - Loads **all** music files from CLEAN dir(s) as clean targets.
    - Loads a long FM noise recording (e.g. superNoiseFM.wav).
    - For each sample, randomly crops a segment of clean + matching
      segment of noise, mixes them at a random SNR and returns
      (noisy, clean).
    """

    def __init__(self, clean_dir, noise_file, audio_length=88192, sample_rate=44100):
        self.clean_dir = clean_dir
        self.noise_file = noise_file
        self.audio_length = audio_length
        self.sample_rate = sample_rate

        print(f"[OnFlyMusicNoise] Scanning clean files in {clean_dir}")
        clean_path = Path(clean_dir)
        self.clean_files = []
        for ext in ("*.flac", "*.wav", "*.mp3"):
            self.clean_files.extend([str(p) for p in clean_path.rglob(ext)])

        if not self.clean_files:
            raise RuntimeError(f"No music files found in {clean_dir}")

        # Shuffle for randomness
        np.random.shuffle(self.clean_files)
        print(f"[OnFlyMusicNoise] Found {len(self.clean_files)} clean music files (no cap)")

        # Load noise file once
        if not Path(noise_file).exists():
            raise FileNotFoundError(f"Noise file not found: {noise_file}")

        print(f"[OnFlyMusicNoise] Loading noise from {noise_file}")
        noise_audio, noise_sr = librosa.load(noise_file, sr=sample_rate)
        if noise_sr != sample_rate:
            print(f"[OnFlyMusicNoise] Resampling noise from {noise_sr} -> {sample_rate}")
            noise_audio = librosa.resample(noise_audio, orig_sr=noise_sr, target_sr=sample_rate)

        if len(noise_audio) < audio_length:
            reps = int(np.ceil(audio_length / len(noise_audio)))
            noise_audio = np.tile(noise_audio, reps)

        self.noise_audio = noise_audio

    def __len__(self):
        return len(self.clean_files)

    def _add_noise_snr(self, clean, noise, snr_db):
        """
        Mix clean + noise at a desired SNR (dB) with proper normalization.
        
        IMPORTANT: Normalizes both signals to standard RMS (0.1) BEFORE calculating SNR.
        This ensures SNR values are consistent regardless of original file volumes.
        Recalculates noise power for each chunk since noise segments vary.
        
        Args:
            clean: Clean audio segment
            noise: Noise audio segment (same length as clean)
            snr_db: Target SNR in decibels (positive = speech louder, negative = noise louder)
        
        Returns:
            Mixed audio at target SNR
        """
        # Step 1: Normalize clean audio to standard RMS = 0.1
        clean_rms = np.sqrt(np.mean(clean ** 2)) + 1e-8
        clean_normalized = clean * (0.1 / clean_rms)
        
        # Step 2: Normalize noise chunk to standard RMS = 0.1
        # CRITICAL: Calculate power for THIS specific noise chunk (not the whole file)
        noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-8
        noise_normalized = noise * (0.1 / noise_rms)
        
        # Step 3: Calculate target noise power based on SNR
        # SNR(dB) = 10*log10(signal_power/noise_power)
        # => noise_power_target = signal_power / (10^(SNR/10))
        clean_power = np.mean(clean_normalized ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        target_noise_power = clean_power / snr_linear
        
        # Step 4: Scale normalized noise to achieve target SNR
        # Recalculate noise power for the normalized chunk
        noise_power = np.mean(noise_normalized ** 2) + 1e-8
        scaling = np.sqrt(target_noise_power / noise_power)
        scaled_noise = noise_normalized * scaling
        
        # Step 5: Mix and return
        return clean_normalized + scaled_noise

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx % len(self.clean_files)]
        clean_audio, sr = librosa.load(clean_path, sr=self.sample_rate)

        if len(clean_audio) < self.audio_length:
            reps = int(np.ceil(self.audio_length / len(clean_audio)))
            clean_audio = np.tile(clean_audio, reps)

        start = np.random.randint(0, len(clean_audio) - self.audio_length + 1)
        clean_seg = clean_audio[start:start + self.audio_length]

        if len(self.noise_audio) < self.audio_length:
            reps = int(np.ceil(self.audio_length / len(self.noise_audio)))
            self.noise_audio = np.tile(self.noise_audio, reps)

        max_start = len(self.noise_audio) - self.audio_length
        n_start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        noise_seg = self.noise_audio[n_start:n_start + self.audio_length]

        snr_db = float(np.random.uniform(5.0, 20.0))
        noisy_seg = self._add_noise_snr(clean_seg, noise_seg, snr_db)

        noisy_tensor = torch.FloatTensor(noisy_seg).unsqueeze(0)
        clean_tensor = torch.FloatTensor(clean_seg).unsqueeze(0)
        return noisy_tensor, clean_tensor


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch with gradient clipping"""
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for noisy, clean in progress_bar:
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for noisy, clean in progress_bar:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            output = model(noisy)
            loss = criterion(output, clean)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    best_epoch = val_losses.index(min(val_losses)) + 1
    best_val_loss = min(val_losses)
    plt.plot(best_epoch, best_val_loss, 'g*', markersize=15, 
             label=f'Best Val Loss ({best_val_loss:.6f})')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Music Denoiser: Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Loss plot saved to: {save_path}")
    
    plt.show()


def train_model(config, resume_from_checkpoint=None):
    """
    Main training function for music denoiser.
    
    Args:
        config: Training configuration
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    print(f"\n{'='*60}")
    print(f"Training Music Audio Denoiser (1D U-Net)")
    print(f"{'='*60}")
    print(f"Device: {config.DEVICE}")
    print(f"Sample Rate: {config.SAMPLE_RATE} Hz (CD quality)")
    print(f"Audio Length: {config.AUDIO_LENGTH} samples")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    if resume_from_checkpoint:
        print(f"🔄 Resuming from: {resume_from_checkpoint}")
    print(f"{'='*60}\n")
    
    # Create dataset: concat of OnFlyMusicNoiseDataset over all CLEAN_AUDIO_DIRS
    from torch.utils.data import ConcatDataset

    datasets = []
    for clean_dir in getattr(config, "CLEAN_AUDIO_DIRS", []):
        datasets.append(
            OnFlyMusicNoiseDataset(
                clean_dir,
                config.NOISE_FILE,
                config.AUDIO_LENGTH,
                config.SAMPLE_RATE,
            )
        )

    if not datasets:
        raise RuntimeError("No CLEAN_AUDIO_DIRS configured for MusicConfig")

    dataset = ConcatDataset(datasets)
    
    # Split into train and validation
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    model = UNet1D(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS
    ).to(config.DEVICE)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer_type = getattr(config, 'OPTIMIZER', 'adam').lower()
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler_patience = getattr(config, 'SCHEDULER_PATIENCE', 5)
    scheduler_factor = getattr(config, 'SCHEDULER_FACTOR', 0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, 
        verbose=True, min_lr=1e-7
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    early_stopping_patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 20)
    epochs_no_improve = 0
    
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"📂 Loading checkpoint from {resume_from_checkpoint}...")
        checkpoint = torch.load(resume_from_checkpoint, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        print(f"✅ Resumed from epoch {start_epoch}")
        print(f"   Best val loss so far: {best_val_loss:.6f}")
        print(f"   Epochs without improvement: {epochs_no_improve}\n")
    
    # Training loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.8f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            Path(config.MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, config.MODEL_SAVE_PATH)
            print(f"✅ Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement for {epochs_no_improve} epoch(s)")
        
        # Save checkpoint
        checkpoint_interval = getattr(config, 'SAVE_CHECKPOINT_EVERY', 1)
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = getattr(config, 'CHECKPOINT_DIR', 'saved_models/music_checkpoints')
            checkpoint_path = Path(checkpoint_dir) / 'latest_checkpoint.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs_no_improve': epochs_no_improve,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved to {checkpoint_path}")
        
        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"{'='*60}\n")
    
    # Plot training history
    plot_save_path = str(Path(config.MODEL_SAVE_PATH).parent / "music_training_history.png")
    plot_training_history(train_losses, val_losses, save_path=plot_save_path)
    
    return train_losses, val_losses, model


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'resume':
            # Resume training
            checkpoint_path = 'saved_models/music_checkpoints/latest_checkpoint.pth'
            if len(sys.argv) > 2:
                checkpoint_path = sys.argv[2]
            
            if not os.path.exists(checkpoint_path):
                print(f"❌ Checkpoint not found: {checkpoint_path}")
                print("Available checkpoints:")
                checkpoint_dir = Path('saved_models/music_checkpoints')
                if checkpoint_dir.exists():
                    for ckpt in checkpoint_dir.glob('*.pth'):
                        print(f"  - {ckpt}")
            else:
                config = MusicConfig()
                train_losses, val_losses, model = train_model(config, resume_from_checkpoint=checkpoint_path)
                print(f"\n✅ Training complete! Model saved to {config.MODEL_SAVE_PATH}")
        else:
            print("Unknown mode. Use: 'resume' or no arguments for fresh training")
    else:
        # Default: train with config
        print("Usage:")
        print("  python backshot_music.py         - Train music model from scratch")
        print("  python backshot_music.py resume  - Resume from checkpoint")
        print()
        
        config = MusicConfig()
        train_losses, val_losses, model = train_model(config)
        print(f"\n✅ Training complete! Model saved to {config.MODEL_SAVE_PATH}")
