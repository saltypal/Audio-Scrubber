"""
System Validation Script

Run this to verify everything is set up correctly before training.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_imports():
    """Test all critical imports"""
    print("\n" + "="*60)
    print("CHECKING IMPORTS")
    print("="*60)
    
    try:
        from config import Paths, AudioSettings, TrainingConfig
        print("✅ config.py imports successfully")
    except Exception as e:
        print(f"❌ config.py import failed: {e}")
        return False
    
    try:
        from config_music import Paths as MusicPaths, AudioSettings as MusicAudio
        print("✅ config_music.py imports successfully")
    except Exception as e:
        print(f"❌ config_music.py import failed: {e}")
        return False
    
    try:
        from src.model.neuralnet import UNet1D
        print("✅ UNet1D model imports successfully")
    except Exception as e:
        print(f"❌ UNet1D import failed: {e}")
        return False
    
    return True


def check_paths():
    """Verify all critical paths exist"""
    print("\n" + "="*60)
    print("CHECKING PATHS")
    print("="*60)
    
    from config import Paths
    
    critical_paths = {
        'LibriSpeech SNR Clean': Paths.LIBRISPEECH_PROCESSED_SNR / 'clean',
        'LibriSpeech SNR Noisy': Paths.LIBRISPEECH_PROCESSED_SNR / 'noisy',
        'Music Clean': Paths.MUSIC_PROCESSED / 'clean',
        'Music Noisy': Paths.MUSIC_PROCESSED / 'noisy',
        'Model Directory': Paths.MODEL_ROOT,
        'Checkpoints': Paths.MODEL_ROOT / 'checkpoints',
        'Music Checkpoints': Paths.MODEL_ROOT / 'music_checkpoints',
    }
    
    all_exist = True
    for name, path in critical_paths.items():
        if path.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (NOT FOUND)")
            all_exist = False
    
    return all_exist


def check_datasets():
    """Check dataset file counts"""
    print("\n" + "="*60)
    print("CHECKING DATASETS")
    print("="*60)
    
    # Check LibriSpeech SNR dataset
    snr_clean = list(Path('dataset/LibriSpeech_processed_snr/clean').rglob('*.flac'))
    snr_noisy = list(Path('dataset/LibriSpeech_processed_snr/noisy').rglob('*.flac'))
    
    print(f"\n📊 LibriSpeech SNR Dataset:")
    print(f"   Clean files: {len(snr_clean)}")
    print(f"   Noisy files: {len(snr_noisy)}")
    
    if len(snr_clean) > 0:
        ratio = len(snr_noisy) / len(snr_clean)
        print(f"   Ratio: {ratio:.1f}x (expected: 6.0x for 6 SNR levels)")
        
        if ratio >= 5.0:
            print("   ✅ Dataset looks good!")
            if snr_noisy:
                print(f"   Sample: {snr_noisy[0].name}")
        else:
            print("   ⚠️  Dataset may be incomplete")
    else:
        print("   ❌ No clean files found! Run AWN2.py to process dataset.")
        return False
    
    # Check music dataset
    music_clean = list(Path('dataset/music_processed/clean').rglob('*.flac'))
    music_noisy = list(Path('dataset/music_processed/noisy').rglob('*.flac'))
    
    print(f"\n🎵 Music Dataset:")
    print(f"   Clean files: {len(music_clean)}")
    print(f"   Noisy files: {len(music_noisy)}")
    
    if len(music_clean) == 0:
        print("   ℹ️  No music files yet (add music to dataset/music/raw and run AWN2.py)")
    
    return True


def check_config():
    """Display current configuration"""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    from config import AudioSettings, TrainingConfig
    
    print(f"\n🗣️  Speech Model Config:")
    print(f"   Sample Rate: {AudioSettings.SAMPLE_RATE} Hz")
    print(f"   Audio Length: {AudioSettings.AUDIO_LENGTH} samples (~{AudioSettings.AUDIO_LENGTH/AudioSettings.SAMPLE_RATE:.1f}s)")
    print(f"   Batch Size: {TrainingConfig.BATCH_SIZE}")
    print(f"   Learning Rate: {TrainingConfig.LEARNING_RATE}")
    print(f"   Epochs: {TrainingConfig.NUM_EPOCHS} (~{TrainingConfig.NUM_EPOCHS * 12} min = ~{TrainingConfig.NUM_EPOCHS * 12 / 60:.1f} hours)")
    print(f"   Device: {TrainingConfig.DEVICE}")
    print(f"   Optimizer: {TrainingConfig.OPTIMIZER}")
    print(f"   Early Stopping Patience: {TrainingConfig.EARLY_STOPPING_PATIENCE} epochs")
    
    from config_music import AudioSettings as MusicAudio, TrainingConfig as MusicTraining
    
    print(f"\n🎵 Music Model Config:")
    print(f"   Sample Rate: {MusicAudio.SAMPLE_RATE} Hz")
    print(f"   Audio Length: {MusicAudio.AUDIO_LENGTH} samples (~{MusicAudio.AUDIO_LENGTH/MusicAudio.SAMPLE_RATE:.1f}s)")
    print(f"   Batch Size: {MusicTraining.BATCH_SIZE}")
    print(f"   Epochs: {MusicTraining.NUM_EPOCHS}")
    
    return True


def check_gpu():
    """Check GPU availability"""
    print("\n" + "="*60)
    print("GPU CHECK")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("ℹ️  CUDA not available - training will use CPU")
        print("   Note: CPU training will be slower (~10x)")
    
    return True


def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("AUDIO DENOISER SYSTEM VALIDATION")
    print("="*60)
    
    checks = [
        ("Imports", check_imports),
        ("Paths", check_paths),
        ("Datasets", check_datasets),
        ("Configuration", check_config),
        ("GPU", check_gpu),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} check failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r for _, r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - READY FOR TRAINING!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Train speech model:")
        print("     python src/model/backshot.py")
        print("\n  2. Train music model (when you have music data):")
        print("     python src/model/backshot_music.py")
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES BEFORE TRAINING")
        print("="*60)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
