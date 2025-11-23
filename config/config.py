"""
Central Configuration File for AudioScrubber Project

This file contains all paths, hyperparameters, and settings used across the project.
Import this in any script to maintain consistency.

Usage:
    from config import Config, Paths, AudioSettings, NoiseSettings
    
    # Use paths
    model_path = Paths.MODEL_SAVE_DIR / "best_model.pth"
    
    # Use settings
    sr = AudioSettings.SAMPLE_RATE
"""

from pathlib import Path
import torch

# ============================================================================
# Base Paths
# ============================================================================

class Paths:
    """All file paths used in the project"""
    
    # Project root (this file's parent directory)
    PROJECT_ROOT = Path(__file__).parent
    
    # Dataset paths
    DATASET_ROOT = PROJECT_ROOT / "dataset"
    
    # LibriSpeech dataset
    LIBRISPEECH_ROOT = DATASET_ROOT / "LibriSpeech"
    LIBRISPEECH_DEV_CLEAN = LIBRISPEECH_ROOT / "dev-clean"
    LIBRISPEECH_DEV_OTHER = LIBRISPEECH_ROOT / "dev-other"
    LIBRISPEECH_PROCESSED = DATASET_ROOT / "LibriSpeech_processed"
    LIBRISPEECH_PROCESSED_SNR = DATASET_ROOT / "LibriSpeech_processed_snr"  # SNR-based noise
    
    # Music dataset
    MUSIC_ROOT = DATASET_ROOT / "music"
    MUSIC_RAW = MUSIC_ROOT / "raw"
    MUSIC_PROCESSED = DATASET_ROOT / "music_processed"
    
    # Instant dataset (smaller, for quick testing)
    INSTANT_ROOT = DATASET_ROOT / "instant"
    INSTANT_CLEAN = INSTANT_ROOT / "clean"
    INSTANT_NOISY = INSTANT_ROOT / "noisy"
    
    # Noise files
    NOISE_ROOT = DATASET_ROOT / "noise"
    STATIC_NOISE = NOISE_ROOT / "static.flac"
    NOISE_PURE = NOISE_ROOT / "FM_Noise.wav"  # Real FM radio noise for training
    
    # Model paths
    MODEL_ROOT = PROJECT_ROOT / "saved_models"
    
    # FM Radio model paths
    MODEL_FM_ROOT = MODEL_ROOT / "FM"
    MODEL_FM_BEST = MODEL_FM_ROOT / "FM_Final_1DUNET" / "music.pth"  # Final trained model
    MODEL_FM_ONFLY_BEST = MODEL_FM_ROOT / "unet1d_onfly_best.pth"
    MODEL_FM_CHECKPOINTS = MODEL_FM_ROOT / "checkpoints"
    
    # Legacy paths (for backward compatibility)
    MODEL_BEST = MODEL_FM_BEST  # Default to FM model
    MODEL_TUNING_DIR = MODEL_ROOT / "tuning"
    
    # Output paths
    OUTPUT_ROOT = PROJECT_ROOT / "output"
    DENOISED_OUTPUT = OUTPUT_ROOT / "denoised"
    
    # Testing paths
    TESTING_ROOT = PROJECT_ROOT / "Tests"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories if they don't exist"""
        dirs_to_create = [
            cls.DATASET_ROOT,
            cls.LIBRISPEECH_ROOT,
            cls.LIBRISPEECH_PROCESSED,
            cls.LIBRISPEECH_PROCESSED_SNR,
            cls.MUSIC_ROOT,
            cls.MUSIC_RAW,
            cls.MUSIC_PROCESSED,
            cls.INSTANT_CLEAN,
            cls.INSTANT_NOISY,
            cls.NOISE_ROOT,
            cls.MODEL_ROOT,
            cls.MODEL_FM_ROOT,
            cls.MODEL_FM_CHECKPOINTS,
            cls.MODEL_TUNING_DIR,
            cls.DENOISED_OUTPUT,
            cls.TESTING_ROOT,
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ All directories created/verified")


# ============================================================================
# Audio Settings
# ============================================================================

class AudioSettings:
    """Audio processing settings"""
    
    # Sample rate for all audio processing
    SAMPLE_RATE = 44100  # Updated to 44.1 kHz for overnight training
    
    # Audio length for training (in samples)
    # Must be divisible by 16 (2^4) for U-Net with 4 downsampling layers
    # Using ~2 seconds: 44096 samples (closest to 44100 that's divisible by 16)
    AUDIO_LENGTH = 88192  # 2 seconds at 44.1 kHz  # ~2 seconds, divisible by 16
    
    # Real-time processing settings
    CHUNK_SIZE = 4096  # Samples per chunk for real-time processing
    LATENCY_MS = (CHUNK_SIZE / SAMPLE_RATE) * 1000  # ~186ms
    
    # Audio format
    AUDIO_FORMAT = 'FLAC'
    
    # Channels
    IN_CHANNELS = 1
    OUT_CHANNELS = 1


# ============================================================================
# Noise Settings
# ============================================================================

class NoiseSettings:
    """Noise addition settings for dataset creation"""
    
    # Standard noise levels (SNR)
    NOISE_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.1]
    
    # Quick testing noise levels (fewer for faster processing)
    QUICK_NOISE_LEVELS = [0.01, 0.05]
    
    # Noise types
    NOISE_TYPE_WHITE = "white"
    NOISE_TYPE_STATIC = "static"


# ============================================================================
# Training Configuration
# ============================================================================

class TrainingConfig:
    """Training hyperparameters and settings"""
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Training hyperparameters
    # For a 6GB GPU, use a smaller batch size to avoid OOM
    BATCH_SIZE = 4
    LEARNING_RATE = 0.0001  # Standard for Adam optimizer on audio tasks
    NUM_EPOCHS = 100  # Full training run  # Increased for deeper training
    
    # Optimizer settings (AdamW generally better for deep learning)
    OPTIMIZER = 'adamw'  # 'adam' or 'adamw' - AdamW has better weight decay
    WEIGHT_DECAY = 0.01  # For AdamW regularization
    
    # Learning rate scheduler (reduce LR when plateauing)
    SCHEDULER_MODE = 'min'
    SCHEDULER_FACTOR = 0.5  # Cut LR in half when no improvement
    SCHEDULER_PATIENCE = 3  # Very aggressive: ~36 min before LR reduction
    SCHEDULER_MIN_LR = 1e-7
    
    # Early stopping (prevents overfitting)
    EARLY_STOPPING_PATIENCE = 6  # Stops after ~72 min of no improvement
    
    # Checkpoint settings (for resume capability)
    SAVE_CHECKPOINT_EVERY = 1  # Save checkpoint every N epochs
    CHECKPOINT_DIR = 'saved_models/FM/checkpoints'
    
    # Gradient clipping
    GRADIENT_CLIP_MAX_NORM = 1.0
    
    # Data split
    TRAIN_SPLIT = 0.8  # 80% train, 20% validation
    
    # DataLoader settings
    NUM_WORKERS = 0  # Set to 0 for Windows compatibility


# ============================================================================
# Hyperparameter Tuning
# ============================================================================

class HyperparameterGrid:
    """Hyperparameter search space for tuning"""
    
    # Full grid search (comprehensive)
    FULL_GRID = {
        'BATCH_SIZE': [8, 16, 32],
        'LEARNING_RATE': [0.00005, 0.0001, 0.0002, 0.0005],
        'NUM_EPOCHS': [75, 100, 150],
        'OPTIMIZER': ['adam', 'adamw'],
        'SCHEDULER_PATIENCE': [8, 12, 15],
        'SCHEDULER_FACTOR': [0.5, 0.3],
        'EARLY_STOPPING_PATIENCE': [20, 25],
    }
    
    # Quick grid search (for testing)
    QUICK_GRID = {
        'BATCH_SIZE': [8, 16],
        'LEARNING_RATE': [0.0001, 0.0005],
        'NUM_EPOCHS': [30, 50],
        'OPTIMIZER': ['adam'],
        'SCHEDULER_PATIENCE': [10],
        'SCHEDULER_FACTOR': [0.5],
        'EARLY_STOPPING_PATIENCE': [15],
    }


# ============================================================================
# RTL-SDR Settings
# ============================================================================

class RTLSDRSettings:
    """RTL-SDR radio settings"""
    
    # Default FM frequency (in Hz)
    FM_FREQUENCY = 101.1e6  # 99.5 MHz
    
    # Sample rate for RTL-SDR
    SDR_SAMPLE_RATE = 44100  # Updated to 44.1 kHz for overnight training
    
    # Audio format from rtl_fm
    RTL_FM_FORMAT = 's16le'  # 16-bit signed little-endian
    
    # Queue sizes for real-time processing
    CAPTURE_QUEUE_SIZE = 10
    AI_QUEUE_SIZE = 10


# ============================================================================
# Model Architecture Settings
# ============================================================================

class ModelSettings:
    """U-Net model architecture settings"""
    
    # U-Net parameters
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    
    # Encoder/Decoder layers
    ENCODER_CHANNELS = [64, 128, 256, 512]
    DECODER_CHANNELS = [256, 128, 64]
    
    # Kernel sizes
    KERNEL_SIZE = 3
    STRIDE = 2


# ============================================================================
# Helper Functions
# ============================================================================

def get_config_summary():
    """Print a summary of all configuration settings"""
    print(f"\n{'='*80}")
    print(f"AudioScrubber Configuration Summary")
    print(f"{'='*80}\n")
    
    print(f"📁 Paths:")
    print(f"  Project Root: {Paths.PROJECT_ROOT}")
    print(f"  Dataset Root: {Paths.DATASET_ROOT}")
    print(f"  Model Best: {Paths.MODEL_BEST}")
    print(f"  LibriSpeech: {Paths.LIBRISPEECH_ROOT}")
    
    print(f"\n🎵 Audio Settings:")
    print(f"  Sample Rate: {AudioSettings.SAMPLE_RATE} Hz")
    print(f"  Audio Length: {AudioSettings.AUDIO_LENGTH} samples ({AudioSettings.AUDIO_LENGTH/AudioSettings.SAMPLE_RATE:.1f}s)")
    print(f"  Chunk Size: {AudioSettings.CHUNK_SIZE} samples")
    print(f"  Latency: {AudioSettings.LATENCY_MS:.1f} ms")
    
    print(f"\n🔊 Noise Settings:")
    print(f"  Noise Levels: {NoiseSettings.NOISE_LEVELS}")
    
    print(f"\n🎓 Training:")
    print(f"  Device: {TrainingConfig.DEVICE}")
    print(f"  Batch Size: {TrainingConfig.BATCH_SIZE}")
    print(f"  Learning Rate: {TrainingConfig.LEARNING_RATE}")
    print(f"  Epochs: {TrainingConfig.NUM_EPOCHS}")
    print(f"  Optimizer: {TrainingConfig.OPTIMIZER}")
    
    print(f"\n📻 RTL-SDR:")
    print(f"  FM Frequency: {RTLSDRSettings.FM_FREQUENCY/1e6:.1f} MHz")
    print(f"  Sample Rate: {RTLSDRSettings.SDR_SAMPLE_RATE} Hz")
    
    print(f"\n{'='*80}\n")


def initialize_project():
    """Initialize the project by creating all necessary directories"""
    print(f"\n{'='*80}")
    print(f"Initializing AudioScrubber Project")
    print(f"{'='*80}\n")
    
    Paths.create_directories()
    get_config_summary()


# ============================================================================
# Backward Compatibility (for existing scripts)
# ============================================================================

class Config:
    """
    Legacy Config class for backward compatibility with existing training scripts.
    Maps to the new configuration structure.
    """
    
    def __init__(self):
        # Paths
        self.CLEAN_AUDIO_DIR = str(Paths.INSTANT_CLEAN)
        self.NOISY_AUDIO_DIR = str(Paths.INSTANT_NOISY)
        self.MODEL_SAVE_PATH = str(Paths.MODEL_BEST)
        
        # Audio settings
        self.SAMPLE_RATE = AudioSettings.SAMPLE_RATE
        self.AUDIO_LENGTH = AudioSettings.AUDIO_LENGTH
        self.IN_CHANNELS = AudioSettings.IN_CHANNELS
        self.OUT_CHANNELS = AudioSettings.OUT_CHANNELS
        
        # Training settings
        self.DEVICE = TrainingConfig.DEVICE
        self.BATCH_SIZE = TrainingConfig.BATCH_SIZE
        self.LEARNING_RATE = TrainingConfig.LEARNING_RATE
        self.NUM_EPOCHS = TrainingConfig.NUM_EPOCHS
        self.TRAIN_SPLIT = TrainingConfig.TRAIN_SPLIT
        
        # Optimizer settings
        self.OPTIMIZER = TrainingConfig.OPTIMIZER
        self.SCHEDULER_PATIENCE = TrainingConfig.SCHEDULER_PATIENCE
        self.SCHEDULER_FACTOR = TrainingConfig.SCHEDULER_FACTOR
        self.EARLY_STOPPING_PATIENCE = TrainingConfig.EARLY_STOPPING_PATIENCE


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Initialize project and show configuration
    initialize_project()
