"""
Music-Specific Configuration for Audio Denoiser

This config is optimized for music denoising (separate from speech model).
Music has different acoustic properties requiring adjusted parameters.

Key Differences from Speech Config:
- Higher sample rate (44100 Hz for full audio quality)
- Longer audio segments (2s @ 44.1kHz = 88192 samples)
- Different dataset paths for music data
- Same U-Net architecture (works for both)

Usage:
    # Option 1: Import alongside main config
    from config_music import MusicPaths, MusicAudioSettings
    
    # Option 2: Use as drop-in replacement (modify backshot.py imports)
    from config_music import Paths, AudioSettings, TrainingConfig
"""

from pathlib import Path
import torch

# ============================================================================
# Music Dataset Paths
# ============================================================================

class MusicPaths:
    """All file paths for music dataset"""
    
    PROJECT_ROOT = Path(__file__).parent
    
    # Music dataset paths
    DATASET_ROOT = PROJECT_ROOT / "dataset"
    MUSIC_ROOT = DATASET_ROOT / "music"
    MUSIC_PROCESSED = DATASET_ROOT / "music_processed"
    MUSIC_CLEAN = MUSIC_PROCESSED / "clean"
    MUSIC_NOISY = MUSIC_PROCESSED / "noisy"
    
    # Noise files (can reuse from speech config)
    NOISE_ROOT = DATASET_ROOT / "noise"
    
    # Model paths (separate from speech model)
    MODEL_ROOT = PROJECT_ROOT / "saved_models"
    MODEL_MUSIC_BEST = MODEL_ROOT / "unet1d_music_best.pth"
    MODEL_MUSIC_CHECKPOINT_DIR = MODEL_ROOT / "music_checkpoints"
    
    # Output paths
    OUTPUT_ROOT = PROJECT_ROOT / "output"
    MUSIC_DENOISED_OUTPUT = OUTPUT_ROOT / "music_denoised"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories for music training"""
        dirs_to_create = [
            cls.MUSIC_ROOT,
            cls.MUSIC_PROCESSED,
            cls.MUSIC_CLEAN,
            cls.MUSIC_NOISY,
            cls.NOISE_ROOT,
            cls.MODEL_ROOT,
            cls.MODEL_MUSIC_CHECKPOINT_DIR,
            cls.MUSIC_DENOISED_OUTPUT,
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Music directories created/verified")


# ============================================================================
# Music Audio Settings
# ============================================================================

class MusicAudioSettings:
    """Audio processing settings optimized for music"""
    
    # Higher sample rate for music (full audio quality)
    SAMPLE_RATE = 44100  # CD quality
    
    # Audio length for training (in samples)
    # Must be divisible by 16 for U-Net with 4 downsampling layers
    # Using 2 seconds: 88192 samples (44096 * 2, divisible by 16)
    AUDIO_LENGTH = 88192  # ~2 seconds @ 44.1kHz, divisible by 16
    
    # Real-time processing settings
    CHUNK_SIZE = 8192  # Larger chunks for music
    LATENCY_MS = (CHUNK_SIZE / SAMPLE_RATE) * 1000  # ~186ms
    
    # Audio format
    AUDIO_FORMAT = 'FLAC'  # Lossless for music
    
    # Channels
    IN_CHANNELS = 1  # Mono (can be expanded to 2 for stereo)
    OUT_CHANNELS = 1


# ============================================================================
# Music Noise Settings
# ============================================================================

class MusicNoiseSettings:
    """Noise addition settings for music dataset"""
    
    # Music-specific noise levels (may need different levels than speech)
    # Music is denser, so slightly different noise profile
    NOISE_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.1]
    
    # Noise types commonly found in music recordings
    NOISE_TYPE_WHITE = "white"
    NOISE_TYPE_PINK = "pink"  # More natural for music (1/f spectrum)
    NOISE_TYPE_TAPE_HISS = "tape_hiss"
    NOISE_TYPE_VINYL_CRACKLE = "vinyl_crackle"


# ============================================================================
# Music Training Configuration
# ============================================================================

class MusicTrainingConfig:
    """Training hyperparameters optimized for music"""
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Training hyperparameters (8-hour window)
    # Note: Larger audio length = fewer batches per epoch = faster epochs
    BATCH_SIZE = 16  # May need to reduce to 8 if GPU memory insufficient
    LEARNING_RATE = 0.0001  # Same as speech (proven)
    NUM_EPOCHS = 40  # 8-hour training window
    
    # Optimizer settings
    OPTIMIZER = 'adamw'
    WEIGHT_DECAY = 0.01
    
    # Learning rate scheduler
    SCHEDULER_MODE = 'min'
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3  # Aggressive for 8-hour window
    SCHEDULER_MIN_LR = 1e-7
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 6
    
    # Checkpoint settings
    SAVE_CHECKPOINT_EVERY = 1
    CHECKPOINT_DIR = str(MusicPaths.MODEL_MUSIC_CHECKPOINT_DIR)
    
    # Gradient clipping
    GRADIENT_CLIP_MAX_NORM = 1.0
    
    # Data split
    TRAIN_SPLIT = 0.8
    
    # DataLoader settings
    NUM_WORKERS = 0  # Windows compatibility


# ============================================================================
# Backward Compatibility (for drop-in replacement)
# ============================================================================

# If you want to use this config as drop-in replacement in backshot.py:
# Just change: from config import Paths, AudioSettings, TrainingConfig
# To: from config_music import Paths, AudioSettings, TrainingConfig

Paths = MusicPaths
AudioSettings = MusicAudioSettings
TrainingConfig = MusicTrainingConfig
NoiseSettings = MusicNoiseSettings


# ============================================================================
# Helper Functions
# ============================================================================

def get_music_config_summary():
    """Print a summary of music configuration"""
    print(f"\n{'='*80}")
    print(f"Music Audio Denoiser Configuration")
    print(f"{'='*80}\n")
    
    print(f"📁 Paths:")
    print(f"  Music Dataset: {MusicPaths.MUSIC_ROOT}")
    print(f"  Processed: {MusicPaths.MUSIC_PROCESSED}")
    print(f"  Model: {MusicPaths.MODEL_MUSIC_BEST}")
    
    print(f"\n🎵 Audio Settings (Music-Optimized):")
    print(f"  Sample Rate: {MusicAudioSettings.SAMPLE_RATE} Hz (CD quality)")
    print(f"  Audio Length: {MusicAudioSettings.AUDIO_LENGTH} samples ({MusicAudioSettings.AUDIO_LENGTH/MusicAudioSettings.SAMPLE_RATE:.1f}s)")
    print(f"  Chunk Size: {MusicAudioSettings.CHUNK_SIZE} samples")
    print(f"  Latency: {MusicAudioSettings.LATENCY_MS:.1f} ms")
    
    print(f"\n🔊 Noise Settings:")
    print(f"  Noise Levels: {MusicNoiseSettings.NOISE_LEVELS}")
    print(f"  Types: White, Pink, Tape Hiss, Vinyl Crackle")
    
    print(f"\n🎓 Training (8-hour window):")
    print(f"  Device: {MusicTrainingConfig.DEVICE}")
    print(f"  Batch Size: {MusicTrainingConfig.BATCH_SIZE}")
    print(f"  Learning Rate: {MusicTrainingConfig.LEARNING_RATE}")
    print(f"  Epochs: {MusicTrainingConfig.NUM_EPOCHS}")
    print(f"  Optimizer: {MusicTrainingConfig.OPTIMIZER}")
    
    print(f"\n⚠️  Key Differences from Speech:")
    print(f"  • Sample rate: 44100 Hz (vs 22050 Hz)")
    print(f"  • Audio length: 88192 samples (vs 44096)")
    print(f"  • More complex frequency content")
    print(f"  • Dense audio (less silence)")
    
    print(f"\n{'='*80}\n")


def initialize_music_project():
    """Initialize music project directories and show config"""
    print(f"\n{'='*80}")
    print(f"Initializing Music Audio Denoiser Project")
    print(f"{'='*80}\n")
    
    MusicPaths.create_directories()
    get_music_config_summary()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    initialize_music_project()
