"""
General Audio Denoiser Training Script (backshot_general.py)

Trains a UNIFIED model on BOTH speech (LibriSpeech) and music datasets.
This creates a general-purpose denoiser that works for both types of audio.

Key Features:
- Combines speech (dev-clean + dev-other) and music datasets
- Sample rate: 44100 Hz (supports both speech and music)
- On-the-fly FM noise mixing with proper normalization
- Same architecture, different training data

Usage:
    python src/fm/model/backshot_general.py

Created for overnight training run.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import sys
import librosa
import numpy as np
from pathlib import Path
from neuralnet import UNet1D
from tqdm import tqdm
import matplotlib.pyplot as plt
import soundfile as sf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from config import Paths, AudioSettings
from config_music import AudioSettings as MusicAudioSettings


class GeneralConfig:
    """Configuration for general model (speech + music)"""
    
    # Paths - BOTH speech and music
    SPEECH_DIRS = [
        str(Paths.LIBRISPEECH_DEV_CLEAN),
        str(Paths.LIBRISPEECH_DEV_OTHER),
    ]
    MUSIC_DIRS = [str(Paths.MUSIC_ROOT)]
    NOISE_FILE = str(Paths.NOISE_ROOT / "superNoiseFM.wav")
    
    # Save to specific folder
    MODEL_SAVE_PATH = "saved_models/general_100_norm/unet1d_best_general.pth"
    CHECKPOINT_DIR = "saved_models/general_100_norm/checkpoints"
    
    # Audio parameters - Use 44.1 kHz to support both speech and music
    SAMPLE_RATE = 44100  # CD quality for both
    AUDIO_LENGTH = 88192  # ~2 seconds at 44.1 kHz
    
    # Training hyperparameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    TRAIN_SPLIT = 0.8
    
    # Model parameters
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class OnFlyGeneralNoiseDataset(Dataset):
    """
    Dataset that mixes BOTH speech and music with FM noise on-the-fly.
    Supports 44.1 kHz sample rate for both audio types.
    """
    
    def __init__(self, audio_dirs, noise_file, audio_length=88192, sample_rate=44100, 
                 snr_db_range=(5, 20), audio_type="general"):
        """
        Args:
            audio_dirs: List of directories with audio files (speech or music)
            noise_file: Path to FM noise file
            audio_length: Target length
            sample_rate: 44100 Hz
            snr_db_range: SNR range for mixing
            audio_type: "speech", "music", or "general" (for logging)
        """
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.snr_db_range = snr_db_range
        self.audio_type = audio_type
        
        # Collect all audio files from all directories
        self.audio_files = []
        for audio_dir in audio_dirs:
            print(f"[{audio_type}] Loading files from {audio_dir}")
            audio_path = Path(audio_dir)
            
            # Support multiple formats
            files = list(audio_path.rglob('*.flac'))
            files += list(audio_path.rglob('*.wav'))
            files += list(audio_path.rglob('*.mp3'))
            
            self.audio_files.extend([str(f) for f in files])
        
        np.random.shuffle(self.audio_files)
        print(f"[{audio_type}] Found {len(self.audio_files)} audio files at {sample_rate} Hz")
        
        # Load FM noise
        print(f"[{audio_type}] Loading FM noise from {noise_file}")
        self.noise_audio, _ = sf.read(noise_file)
        
        # Convert to mono if stereo
        if len(self.noise_audio.shape) > 1:
            self.noise_audio = np.mean(self.noise_audio, axis=1)
        
        # Tile noise to have enough samples
        if len(self.noise_audio) < audio_length * 10:
            repeats = (audio_length * 10) // len(self.noise_audio) + 1
            self.noise_audio = np.tile(self.noise_audio, repeats)
        
        print(f"[{audio_type}] Noise loaded: {len(self.noise_audio)} samples")
        print(f"[{audio_type}] SNR range: {snr_db_range[0]} to {snr_db_range[1]} dB")
    
    def __len__(self):
        return len(self.audio_files)
    
    def _add_noise_snr(self, clean_audio, snr_db):
        """
        Add FM noise with proper normalization (consistent SNR).
        """
        # Step 1: Get random noise segment
        if len(self.noise_audio) > len(clean_audio):
            start_idx = np.random.randint(0, len(self.noise_audio) - len(clean_audio))
            noise_segment = self.noise_audio[start_idx:start_idx + len(clean_audio)]
        else:
            noise_segment = self.noise_audio[:len(clean_audio)]
        
        # Step 2: Normalize clean audio to RMS = 0.1
        clean_rms = np.sqrt(np.mean(clean_audio ** 2)) + 1e-8
        clean_normalized = clean_audio * (0.1 / clean_rms)
        
        # Step 3: Normalize noise chunk to RMS = 0.1
        noise_rms = np.sqrt(np.mean(noise_segment ** 2)) + 1e-8
        noise_normalized = noise_segment * (0.1 / noise_rms)
        
        # Step 4: Calculate target noise power based on SNR
        signal_power = np.mean(clean_normalized ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power_target = signal_power / snr_linear
        
        # Step 5: Scale normalized noise
        noise_power = np.mean(noise_normalized ** 2) + 1e-8
        noise_scale = np.sqrt(noise_power_target / noise_power)
        scaled_noise = noise_normalized * noise_scale
        
        # Step 6: Mix
        noisy_audio = clean_normalized + scaled_noise
        
        return noisy_audio
    
    def __getitem__(self, idx):
        """Load audio and add FM noise on-the-fly."""
        # Load audio file
        audio_path = self.audio_files[idx]
        clean_audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # Random SNR
        snr_db = np.random.uniform(self.snr_db_range[0], self.snr_db_range[1])
        
        # Ensure correct length
        if len(clean_audio) > self.audio_length:
            start = np.random.randint(0, len(clean_audio) - self.audio_length)
            clean_audio = clean_audio[start:start + self.audio_length]
        elif len(clean_audio) < self.audio_length:
            pad_length = self.audio_length - len(clean_audio)
            clean_audio = np.pad(clean_audio, (0, pad_length), mode='constant')
        
        # Add noise
        noisy_audio = self._add_noise_snr(clean_audio, snr_db)
        
        # Safety check
        clean_audio = clean_audio[:self.audio_length]
        noisy_audio = noisy_audio[:self.audio_length]
        
        # Convert to tensors
        noisy_tensor = torch.FloatTensor(noisy_audio).unsqueeze(0)
        clean_tensor = torch.FloatTensor(clean_audio).unsqueeze(0)
        
        return noisy_tensor, clean_tensor


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
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
    """Validate the model."""
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
    """Plot and save training history."""
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
    plt.title('General Model Training History (Speech + Music)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Loss plot saved to: {save_path}")
    
    plt.close()


def train_general_model():
    """Main training function for general model."""
    config = GeneralConfig()
    
    print(f"\n{'='*70}")
    print(f"Training GENERAL Model (Speech + Music)")
    print(f"{'='*70}")
    print(f"Device: {config.DEVICE}")
    print(f"Sample Rate: {config.SAMPLE_RATE} Hz")
    print(f"Audio Length: {config.AUDIO_LENGTH} samples")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"Model save path: {config.MODEL_SAVE_PATH}")
    print(f"{'='*70}\n")
    
    # Create datasets for BOTH speech and music
    print("[INFO] Creating datasets for speech and music...")
    datasets = []
    
    # Add speech datasets
    for speech_dir in config.SPEECH_DIRS:
        datasets.append(OnFlyGeneralNoiseDataset(
            [speech_dir],
            config.NOISE_FILE,
            config.AUDIO_LENGTH,
            config.SAMPLE_RATE,
            snr_db_range=(5, 20),
            audio_type="speech"
        ))
    
    # Add music datasets
    for music_dir in config.MUSIC_DIRS:
        datasets.append(OnFlyGeneralNoiseDataset(
            [music_dir],
            config.NOISE_FILE,
            config.AUDIO_LENGTH,
            config.SAMPLE_RATE,
            snr_db_range=(5, 20),
            audio_type="music"
        ))
    
    # Combine all datasets
    dataset = ConcatDataset(datasets)
    print(f"\n[INFO] Total combined dataset size: {len(dataset)} samples")
    
    # Split into train and validation
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
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
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs_no_improve = 0
    early_stopping_patience = 20
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, config.DEVICE)
        val_losses.append(val_loss)
        
        # Update LR
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
            print(f"[OK] Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
        
        # Save checkpoint
        checkpoint_path = Path(config.CHECKPOINT_DIR) / 'latest_checkpoint.pth'
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
        
        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n[WARNING] Early stopping at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"{'='*70}\n")
    
    # Plot history
    plot_save_path = str(Path(config.MODEL_SAVE_PATH).parent / "training_history.png")
    plot_training_history(train_losses, val_losses, save_path=plot_save_path)
    
    return train_losses, val_losses, model


if __name__ == "__main__":
    train_general_model()
