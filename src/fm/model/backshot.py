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
import soundfile as sf

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import Paths, AudioSettings, TrainingConfig

"""
Created by Satya with Copilot @ 15/11/25

Training script for 1D U-Net Audio Denoiser
- Loads clean and noisy audio pairs
- Trains the model to predict clean audio from noisy input
- Saves the best model
- Supports hyperparameter tuning
"""

# --- Configuration ---
class Config:
    # Paths - Using LibriSpeech processed dataset (dev-clean)
    CLEAN_AUDIO_DIR = str(Paths.LIBRISPEECH_DEV_CLEAN)
    NOISE_FILE = str(Paths.NOISE_PURE)  # Real FM noise recording
    MODEL_SAVE_PATH = str(Paths.MODEL_FM_BEST)
    CHECKPOINT_DIR = str(Paths.MODEL_FM_CHECKPOINTS)
    
    # Audio parameters - Using config
    SAMPLE_RATE = AudioSettings.SAMPLE_RATE
    AUDIO_LENGTH = AudioSettings.AUDIO_LENGTH
    
    # Training hyperparameters - Using config defaults
    BATCH_SIZE = TrainingConfig.BATCH_SIZE
    NUM_EPOCHS = TrainingConfig.NUM_EPOCHS
    LEARNING_RATE = TrainingConfig.LEARNING_RATE
    TRAIN_SPLIT = TrainingConfig.TRAIN_SPLIT
    
    # Model parameters
    IN_CHANNELS = AudioSettings.IN_CHANNELS
    OUT_CHANNELS = AudioSettings.OUT_CHANNELS
    
    # Device
    DEVICE = TrainingConfig.DEVICE


class AudioDataset(Dataset):
    """
    Dataset for loading clean and noisy audio pairs.
    
    Each sample is a pair of (noisy_audio, clean_audio)
    Both are tensors of shape (1, AUDIO_LENGTH)
    """
    def __init__(self, clean_dir, noisy_dir, audio_length=16000, sample_rate=22050):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        print("Loading the clean and noisy files")

        # Get all clean files (recursively)
        clean_path = Path(clean_dir)
        self.clean_files = sorted([str(f.relative_to(clean_path)) for f in clean_path.rglob('*.flac')])
        
        # Get all noisy files (recursively)
        noisy_path = Path(noisy_dir)
        self.noisy_files = sorted([str(f.relative_to(noisy_path)) for f in noisy_path.rglob('*.flac')])
        
        print(f"Found {len(self.clean_files)} clean files") 
        print(f"Found {len(self.noisy_files)} noisy files")
    
    def __len__(self):
        # Return the number of noisy files (we have multiple noisy versions per clean file)
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        """
        Load a noisy audio file and its corresponding clean version.
        Ensures exact audio_length for all samples to prevent tensor size mismatches.
        """
        # Load noisy audio (use full path)
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        noisy_audio, _ = librosa.load(noisy_path, sr=self.sample_rate)
        
        # Extract the base filename to find the corresponding clean file
        # Noisy filename format with SNR: "1272-128104-0000_snr_10dB.flac"
        # Clean filename format: "1272-128104-0000.flac"
        noisy_filename = Path(self.noisy_files[idx]).name
        
        # Remove the SNR suffix to get clean filename
        if '_snr_' in noisy_filename:
            clean_filename = noisy_filename.split('_snr_')[0] + '.flac'
        else:
            # Fallback for old format
            if '_noise_level_' in noisy_filename:
                clean_filename = noisy_filename.split('_noise_level_')[0] + '.flac'
            else:
                clean_filename = noisy_filename
        
        # Get the directory structure from noisy file and apply to clean
        noisy_rel_dir = Path(self.noisy_files[idx]).parent
        clean_rel_path = noisy_rel_dir / clean_filename
        
        # Load clean audio
        clean_path = os.path.join(self.clean_dir, str(clean_rel_path))
        clean_audio, _ = librosa.load(clean_path, sr=self.sample_rate)
        
        # Ensure both have the same length first
        min_length = min(len(noisy_audio), len(clean_audio))
        noisy_audio = noisy_audio[:min_length]
        clean_audio = clean_audio[:min_length]
        
        # CRITICAL: Ensure exact audio_length to prevent tensor size mismatches
        # Always truncate first, then pad if needed
        if len(noisy_audio) > self.audio_length:
            # Truncate to exact length
            noisy_audio = noisy_audio[:self.audio_length]
            clean_audio = clean_audio[:self.audio_length]
        elif len(noisy_audio) < self.audio_length:
            # Pad to exact length
            pad_length = self.audio_length - len(noisy_audio)
            noisy_audio = np.pad(noisy_audio, (0, pad_length), mode='constant')
            clean_audio = np.pad(clean_audio, (0, pad_length), mode='constant')
        
        # Double check: force exact length (safety measure)
        noisy_audio = noisy_audio[:self.audio_length]
        clean_audio = clean_audio[:self.audio_length]
        
        # Verify lengths match exactly
        assert len(noisy_audio) == self.audio_length, f"Noisy audio length mismatch: {len(noisy_audio)} != {self.audio_length}"
        assert len(clean_audio) == self.audio_length, f"Clean audio length mismatch: {len(clean_audio)} != {self.audio_length}"
        
        # Convert to tensors and add channel dimension
        noisy_tensor = torch.FloatTensor(noisy_audio).unsqueeze(0)  # (1, audio_length)
        clean_tensor = torch.FloatTensor(clean_audio).unsqueeze(0)  # (1, audio_length)
        
        return noisy_tensor, clean_tensor


class OnFlyNoiseDataset(Dataset):
    """
    Dataset that generates noisy audio ON-THE-FLY by adding real FM noise
    to clean LibriSpeech audio using SNR (Signal-to-Noise Ratio) method.
    
    This approach:
    - Uses REAL recorded FM noise (dataset/noise/pure10noise.wav)
    - Adds noise dynamically during training (no pre-generated files needed)
    - Uses SNR to control noise level (more realistic than fixed noise levels)
    - Saves disk space (no need to store noisy versions)
    """
    
    def __init__(self, clean_dir, noise_file, audio_length=44096, sample_rate=22050, 
                 snr_db_range=(5, 20)):
        """
        Args:
            clean_dir: Directory with clean FLAC files (LibriSpeech dev-clean)
            noise_file: Path to recorded FM noise WAV file
            audio_length: Target length for all samples
            sample_rate: Sample rate for audio
            snr_db_range: Tuple (min_snr, max_snr) in dB for random SNR selection
        """
        self.clean_dir = clean_dir
        self.audio_length = audio_length
        self.sample_rate = sample_rate
        self.snr_db_range = snr_db_range
        
        print(f"[OnFlyNoise] Loading clean files from {clean_dir}")
        
        # Get all clean files recursively
        clean_path = Path(clean_dir)
        self.clean_files = sorted([str(f) for f in clean_path.rglob('*.flac')])
        
        print(f"[OnFlyNoise] Found {len(self.clean_files)} clean files")
        
        # Load the real FM noise recording
        print(f"[OnFlyNoise] Loading FM noise from {noise_file}")
        self.noise_audio, _ = sf.read(noise_file)
        
        # Convert to mono if stereo
        if len(self.noise_audio.shape) > 1:
            self.noise_audio = np.mean(self.noise_audio, axis=1)
        
        # Resample noise if needed
        if self.noise_audio.shape[0] < audio_length * 10:
            # Repeat noise to have enough samples
            repeats = (audio_length * 10) // len(self.noise_audio) + 1
            self.noise_audio = np.tile(self.noise_audio, repeats)
        
        print(f"[OnFlyNoise] Noise loaded: {len(self.noise_audio)} samples")
        print(f"[OnFlyNoise] SNR range: {snr_db_range[0]} to {snr_db_range[1]} dB")
    
    def __len__(self):
        return len(self.clean_files)
    
    def _add_noise_snr(self, clean_audio, snr_db):
        """
        Add real FM noise to clean audio using SNR method.
        
        SNR (dB) = 10 * log10(Power_signal / Power_noise)
        Power = mean(signal^2)
        
        Args:
            clean_audio: Clean audio signal
            snr_db: Target SNR in decibels
            
        Returns:
            Noisy audio with specified SNR
        """
        # Calculate signal power
        signal_power = np.mean(clean_audio ** 2)
        
        # Get random segment of noise
        if len(self.noise_audio) > len(clean_audio):
            start_idx = np.random.randint(0, len(self.noise_audio) - len(clean_audio))
            noise_segment = self.noise_audio[start_idx:start_idx + len(clean_audio)]
        else:
            noise_segment = self.noise_audio[:len(clean_audio)]
        
        # Calculate noise power
        noise_power = np.mean(noise_segment ** 2)
        
        # Calculate required noise scaling factor from SNR
        # SNR(dB) = 10*log10(signal_power/noise_power)
        # Therefore: noise_power_target = signal_power / (10^(SNR/10))
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power_target = signal_power / snr_linear
        
        # Scale noise to achieve target SNR
        if noise_power > 0:
            noise_scale = np.sqrt(noise_power_target / noise_power)
        else:
            noise_scale = 0
        
        scaled_noise = noise_segment * noise_scale
        
        # Add noise to signal
        noisy_audio = clean_audio + scaled_noise
        
        return noisy_audio
    
    def __getitem__(self, idx):
        """
        Load clean audio and add real FM noise on-the-fly.
        """
        # Load clean audio
        clean_path = self.clean_files[idx]
        clean_audio, _ = librosa.load(clean_path, sr=self.sample_rate)
        
        # Randomly select SNR from range
        snr_db = np.random.uniform(self.snr_db_range[0], self.snr_db_range[1])
        
        # Ensure audio length matches
        if len(clean_audio) > self.audio_length:
            # Random crop for data augmentation
            start = np.random.randint(0, len(clean_audio) - self.audio_length)
            clean_audio = clean_audio[start:start + self.audio_length]
        elif len(clean_audio) < self.audio_length:
            # Pad with zeros
            pad_length = self.audio_length - len(clean_audio)
            clean_audio = np.pad(clean_audio, (0, pad_length), mode='constant')
        
        # Add real FM noise using SNR method
        noisy_audio = self._add_noise_snr(clean_audio, snr_db)
        
        # Ensure exact length (safety check)
        clean_audio = clean_audio[:self.audio_length]
        noisy_audio = noisy_audio[:self.audio_length]
        
        # Convert to tensors
        noisy_tensor = torch.FloatTensor(noisy_audio).unsqueeze(0)  # (1, audio_length)
        clean_tensor = torch.FloatTensor(clean_audio).unsqueeze(0)  # (1, audio_length)
        
        return noisy_tensor, clean_tensor


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch with gradient clipping.
    
    Returns:
        Average training loss for this epoch
    """
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for noisy, clean in progress_bar:
        # Move data to device
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(noisy)
        
        # Calculate loss
        loss = criterion(output, clean)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for noisy, clean in progress_bar:
            # Move data to device
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # Forward pass
            output = model(noisy)
            
            # Calculate loss
            loss = criterion(output, clean)
            
            # Track loss
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def plot_training_history(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save the plot image
    """
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    # Mark best validation loss
    best_epoch = val_losses.index(min(val_losses)) + 1
    best_val_loss = min(val_losses)
    plt.plot(best_epoch, best_val_loss, 'g*', markersize=15, label=f'Best Val Loss ({best_val_loss:.6f})')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
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
    Main training function with checkpoint resume capability.
    
    Args:
        config: Training configuration
        resume_from_checkpoint: Path to checkpoint file to resume from (optional)
    """
    print(f"\n{'='*60}")
    print(f"Training 1D U-Net Audio Denoiser")
    print(f"{'='*60}")
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    if resume_from_checkpoint:
        print(f"🔄 Resuming from: {resume_from_checkpoint}")
    print(f"{'='*60}\n")
    
    # Create dataset - Using ON-THE-FLY noise generation with real FM noise
    print("\n[INFO] Using OnFlyNoiseDataset with real FM noise")
    dataset = OnFlyNoiseDataset(
        config.CLEAN_AUDIO_DIR,
        config.NOISE_FILE,
        config.AUDIO_LENGTH,
        config.SAMPLE_RATE,
        snr_db_range=(5, 20)  # SNR between 5dB (very noisy) and 20dB (less noisy)
    )
    
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
        num_workers=0  # Set to 0 for Windows compatibility
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
    
    # Loss function (Mean Squared Error for audio reconstruction)
    criterion = nn.MSELoss()
    
    # Optimizer (support multiple types)
    optimizer_type = getattr(config, 'OPTIMIZER', 'adam').lower()
    if optimizer_type == 'adamw':
        # AdamW has better weight decay regularization
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler (reduces LR when validation loss plateaus)
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
    
    # Training loop with early stopping
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
        
        # Get current learning rate
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
        
        # Save checkpoint every epoch (for resume capability)
        checkpoint_interval = getattr(config, 'SAVE_CHECKPOINT_EVERY', 1)
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = getattr(config, 'CHECKPOINT_DIR', 'saved_models/FM/checkpoints')
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
    plot_save_path = str(Path(config.MODEL_SAVE_PATH).parent / "training_history.png")
    plot_training_history(train_losses, val_losses, save_path=plot_save_path)
    
    return train_losses, val_losses, model


# --- Hyperparameter Tuning ---
import itertools
import json

# Expanded hyperparameter search space for superior model performance
HYPERPARAM_GRID = {
    'BATCH_SIZE': [8, 16, 32],  # Larger batches for more stable gradients
    'LEARNING_RATE': [0.00005, 0.0001, 0.0002, 0.0005],  # Focused on effective range
    'NUM_EPOCHS': [75, 100, 150],  # More epochs for convergence
    'OPTIMIZER': ['adam', 'adamw'],  # AdamW often performs better with weight decay
    'SCHEDULER_PATIENCE': [8, 12, 15],  # More patient before reducing LR
    'SCHEDULER_FACTOR': [0.5, 0.3],  # LR reduction factors
    'EARLY_STOPPING_PATIENCE': [20, 25],  # Prevent early premature stopping
}

# For quick testing (much faster, fewer combinations)
QUICK_HYPERPARAM_GRID = {
    'BATCH_SIZE': [8, 16],
    'LEARNING_RATE': [0.0001, 0.0005],
    'NUM_EPOCHS': [30, 50],
    'OPTIMIZER': ['adam'],
    'SCHEDULER_PATIENCE': [10],
    'SCHEDULER_FACTOR': [0.5],
    'EARLY_STOPPING_PATIENCE': [15],
}


def train_with_hyperparameter_tuning(quick_mode=False):
    """
    Advanced hyperparameter tuning with intelligent search.
    
    Features:
    - Expanded search space with optimizer types, LR scheduling, early stopping
    - Quick mode for faster experimentation (reduced grid)
    - Comprehensive result tracking with overfitting detection
    - Automatic best model selection and final training
    
    Args:
        quick_mode: If True, uses QUICK_HYPERPARAM_GRID for faster testing
    
    Steps:
    1. Tests all hyperparameter combinations
    2. Tracks performance metrics for each
    3. Finds best configuration
    4. Trains final model with best parameters
    5. Saves results and model
    
    Returns:
        Best model, training history, and best hyperparameters
    """
    print(f"\n{'='*60}")
    print(f"Advanced Hyperparameter Tuning for 1D U-Net")
    print(f"{'='*60}\n")
    
    # Choose hyperparameter grid
    param_grid = QUICK_HYPERPARAM_GRID if quick_mode else HYPERPARAM_GRID
    
    if quick_mode:
        print("🚀 QUICK MODE: Using reduced grid for faster testing\n")
    else:
        print("🔬 FULL MODE: Using complete grid for best results\n")
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to test: {len(combinations)}")
    print(f"This may take several hours depending on hardware.\n")
    
    results = []
    best_val_loss = float('inf')
    best_config = None
    best_result = None
    
    for i, params in enumerate(combinations, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(combinations)}")
        print(f"{'='*60}")
        print(f"Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
        
        # Create config with these hyperparameters
        config = Config()
        for key, value in params.items():
            setattr(config, key, value)
        
        # Update model save path to include key hyperparameters
        opt_suffix = params['OPTIMIZER']
        sp_suffix = params['SCHEDULER_PATIENCE']
        model_name = f"unet1d_bs{params['BATCH_SIZE']}_lr{params['LEARNING_RATE']}_ep{params['NUM_EPOCHS']}_{opt_suffix}_sp{sp_suffix}.pth"
        config.MODEL_SAVE_PATH = f"saved_models/tuning/{model_name}"
        
        try:
            # Train model
            train_losses, val_losses, model = train_model(config)
            
            # Get the best validation loss from this run
            min_val_loss = min(val_losses)
            
            # Record comprehensive results with overfitting analysis
            result = {
                'experiment': i,
                'hyperparameters': params,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'best_val_loss': min_val_loss,
                'epochs_completed': len(train_losses),
                'train_val_gap': abs(train_losses[-1] - val_losses[-1]),  # Overfitting indicator
                'model_path': config.MODEL_SAVE_PATH
            }
            results.append(result)
            
            print(f"\n✅ Experiment {i} completed!")
            print(f"   Final validation loss: {result['final_val_loss']:.6f}")
            print(f"   Best validation loss: {result['best_val_loss']:.6f}")
            print(f"   Epochs completed: {result['epochs_completed']}/{params['NUM_EPOCHS']}")
            print(f"   Train-Val gap: {result['train_val_gap']:.6f}")
            
            # Track overall best
            if min_val_loss < best_val_loss:
                best_val_loss = min_val_loss
                best_config = params
                best_result = result
                print(f"   ⭐ NEW BEST MODEL!")
            
        except Exception as e:
            print(f"\n❌ Experiment {i} failed: {e}")
            results.append({
                'experiment': i,
                'hyperparameters': params,
                'error': str(e)
            })
    
    # Save results
    results_path = "saved_models/tuning/hyperparameter_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Hyperparameter Tuning Completed!")
    print(f"{'='*60}\n")
    
    # Display best configuration
    if best_result:
        print("🏆 Best Configuration Found:")
        print(f"   Experiment: {best_result['experiment']}")
        print(f"   Best validation loss: {best_result['best_val_loss']:.6f}")
        print(f"   Hyperparameters:")
        for key, value in best_result['hyperparameters'].items():
            print(f"     {key}: {value}")
        print(f"   Tuning model saved to: {best_result['model_path']}")
        
        # Train final model with best hyperparameters
        print(f"\n{'='*60}")
        print(f"Training Final Model with Best Hyperparameters")
        print(f"{'='*60}\n")
        
        final_config = Config()
        for key, value in best_config.items():
            setattr(final_config, key, value)
        
        # Save to default location
        final_config.MODEL_SAVE_PATH = r"saved_models\unet1d_best.pth"
        
        train_losses, val_losses, final_model = train_model(final_config)
        
        print(f"\n{'='*60}")
        print(f"✅ Final Model Training Complete!")
        print(f"   Model saved to: {final_config.MODEL_SAVE_PATH}")
        print(f"   Best hyperparameters used:")
        for key, value in best_config.items():
            print(f"     {key}: {value}")
        print(f"{'='*60}\n")
        
        # Save best hyperparameters to file
        best_params_path = "saved_models/best_hyperparameters.json"
        with open(best_params_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"Best hyperparameters saved to: {best_params_path}\n")
        
        return final_model, train_losses, val_losses, best_config
    else:
        print("❌ No successful experiments found!")
        return None, None, None, None
    
    print(f"\nAll results saved to: {results_path}")
    print(f"{'='*60}\n")


def quick_tune():
    """
    Quick hyperparameter tuning with fewer combinations.
    Good for testing before full grid search.
    """
    return train_with_hyperparameter_tuning(quick_mode=True)


def analyze_tuning_results(results_path="saved_models/tuning/hyperparameter_results.json"):
    """
    Analyze and display hyperparameter tuning results.
    
    Shows:
    - Top 5 best configurations
    - Impact of each hyperparameter
    - Overfitting analysis
    
    Args:
        results_path: Path to the JSON file with tuning results
    """
    import json
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Filter out failed experiments
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        print("❌ No successful experiments found!")
        return
    
    # Sort by best validation loss
    successful_results.sort(key=lambda x: x['best_val_loss'])
    
    print(f"\n{'='*80}")
    print(f"Hyperparameter Tuning Results Analysis")
    print(f"{'='*80}\n")
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(results) - len(successful_results)}\n")
    
    # Display top 5 configurations
    print(f"{'='*80}")
    print(f"Top 5 Best Configurations")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(successful_results[:5], 1):
        print(f"{i}. Experiment #{result['experiment']}")
        print(f"   Best Val Loss: {result['best_val_loss']:.6f}")
        print(f"   Train-Val Gap: {result['train_val_gap']:.6f}")
        print(f"   Epochs: {result['epochs_completed']}")
        print(f"   Hyperparameters:")
        for key, value in result['hyperparameters'].items():
            print(f"      • {key}: {value}")
        print()
    
    # Analyze hyperparameter impact
    print(f"{'='*80}")
    print(f"Hyperparameter Impact Analysis")
    print(f"{'='*80}\n")
    
    # Group by each hyperparameter
    hyperparam_keys = successful_results[0]['hyperparameters'].keys()
    
    for key in hyperparam_keys:
        print(f"\n{key}:")
        
        # Group results by this hyperparameter value
        groups = {}
        for result in successful_results:
            value = str(result['hyperparameters'][key])
            if value not in groups:
                groups[value] = []
            groups[value].append(result['best_val_loss'])
        
        # Calculate average for each value
        for value, losses in sorted(groups.items()):
            avg_loss = sum(losses) / len(losses)
            print(f"   {value}: avg loss = {avg_loss:.6f} (n={len(losses)})")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train 1D U-Net Audio Denoiser')
    parser.add_argument('mode', nargs='?', default='train', 
                       choices=['train', 'resume', 'quick', 'full', 'analyze'],
                       help='Training mode: train (default), resume, quick, full, analyze')
    parser.add_argument('--epochs', '-e', type=int, default=None,
                       help='Number of epochs to train (overrides config default)')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                       help='Batch size (overrides config default)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=None,
                       help='Learning rate (overrides config default)')
    parser.add_argument('--checkpoint', '-c', type=str, default='saved_models/FM/checkpoints/latest_checkpoint.pth',
                       help='Checkpoint path for resume mode')
    
    args = parser.parse_args()
    mode = args.mode.lower()
    
    if mode == 'analyze':
        # Analyze existing tuning results
        analyze_tuning_results()
    elif mode == 'quick':
        # Quick tuning mode
        print("Starting quick hyperparameter tuning...")
        quick_tune()
    elif mode == 'full':
        # Full tuning mode
        print("Starting full hyperparameter tuning...")
        train_with_hyperparameter_tuning(quick_mode=False)
    elif mode == 'resume':
        # Resume training from checkpoint
        checkpoint_path = args.checkpoint
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("Available checkpoints:")
            checkpoint_dir = Path('saved_models/FM/checkpoints')
            if checkpoint_dir.exists():
                for ckpt in checkpoint_dir.glob('*.pth'):
                    print(f"  - {ckpt}")
        else:
            config = Config()
            
            # Apply command-line overrides
            if args.epochs is not None:
                config.NUM_EPOCHS = args.epochs
                print(f"🔧 Epochs overridden to: {args.epochs}")
            if args.batch_size is not None:
                config.BATCH_SIZE = args.batch_size
                print(f"🔧 Batch size overridden to: {args.batch_size}")
            if args.learning_rate is not None:
                config.LEARNING_RATE = args.learning_rate
                print(f"🔧 Learning rate overridden to: {args.learning_rate}")
            
            train_losses, val_losses, model = train_model(config, resume_from_checkpoint=checkpoint_path)
            print(f"\n✅ Training complete! Model saved to {config.MODEL_SAVE_PATH}")
    elif mode == 'train':
        # Default: train with config (with optional overrides)
        config = Config()
        
        # Apply command-line overrides
        if args.epochs is not None:
            config.NUM_EPOCHS = args.epochs
            print(f"🔧 Epochs overridden to: {args.epochs}")
        if args.batch_size is not None:
            config.BATCH_SIZE = args.batch_size
            print(f"🔧 Batch size overridden to: {args.batch_size}")
        if args.learning_rate is not None:
            config.LEARNING_RATE = args.learning_rate
            print(f"🔧 Learning rate overridden to: {args.learning_rate}")
        
        train_losses, val_losses, model = train_model(config)
        print(f"\n✅ Training complete! Model saved to {config.MODEL_SAVE_PATH}")
    else:
        print("Unknown mode. Use: 'train', 'quick', 'full', 'analyze', or 'resume'")
