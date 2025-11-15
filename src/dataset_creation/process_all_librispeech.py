import os
import sys
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Paths, AudioSettings, NoiseSettings

"""
Process ALL audio files in the LibriSpeech dataset and add noise.

This script:
1. Recursively finds all .flac files in the LibriSpeech dataset
2. Copies clean files to dataset/LibriSpeech_processed/clean/
3. Adds white noise at multiple levels to each file
4. Saves noisy versions to dataset/LibriSpeech_processed/noisy/

Created to process the entire LibriSpeech dataset for comprehensive training.
"""

# Import paths from central config
LIBRISPEECH_ROOT = Paths.LIBRISPEECH_DEV_CLEAN
CLEAN_OUTPUT_DIR = Paths.LIBRISPEECH_PROCESSED / "dev-clean" / "clean"
NOISY_OUTPUT_DIR = Paths.LIBRISPEECH_PROCESSED / "dev-clean" / "noisy"

# Import settings from central config
NOISE_LEVELS = NoiseSettings.NOISE_LEVELS
TARGET_SAMPLE_RATE = AudioSettings.SAMPLE_RATE


def add_white_noise(audio, noise_level):
    """
    Add white noise to audio signal.
    
    Args:
        audio: Input audio as numpy array
        noise_level: Standard deviation of Gaussian noise
    
    Returns:
        Noisy audio clipped to [-1.0, 1.0]
    """
    noise = np.random.normal(0, noise_level, len(audio))
    noisy_audio = audio + noise
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    return noisy_audio


def find_all_audio_files(root_dir):
    """
    Recursively find all .flac files in the LibriSpeech directory structure.
    
    LibriSpeech structure:
    dev-clean/
        speaker_id/
            chapter_id/
                speaker_id-chapter_id-utterance_id.flac
    
    Args:
        root_dir: Root directory of LibriSpeech subset
    
    Returns:
        List of tuples: (full_path, relative_path)
    """
    audio_files = []
    
    for root, dirs, files in os.walk(str(root_dir)):
        for file in files:
            if file.endswith('.flac'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, str(root_dir))
                audio_files.append((full_path, relative_path))
    
    return audio_files


def process_all_audio_files():
    """
    Main processing function:
    1. Find all .flac files in LibriSpeech
    2. Copy clean versions to output directory
    3. Generate noisy versions at multiple noise levels
    4. Maintain directory structure for organization
    """
    
    print(f"\n{'='*80}")
    print(f"LibriSpeech Dataset Noise Addition")
    print(f"{'='*80}\n")
    
    # Create output directories
    Path(CLEAN_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(NOISY_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"Source: {LIBRISPEECH_ROOT}")
    print(f"Clean output: {CLEAN_OUTPUT_DIR}")
    print(f"Noisy output: {NOISY_OUTPUT_DIR}")
    print(f"Target sample rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"Noise levels: {NOISE_LEVELS}\n")
    
    # Find all audio files
    print("Scanning for audio files...")
    audio_files = find_all_audio_files(LIBRISPEECH_ROOT)
    
    if not audio_files:
        print(f"❌ No audio files found in {LIBRISPEECH_ROOT}")
        return
    
    print(f"✅ Found {len(audio_files)} audio files\n")
    
    # Process each audio file
    total_files = len(audio_files)
    successful = 0
    failed = 0
    
    print(f"{'='*80}")
    print(f"Processing Files")
    print(f"{'='*80}\n")
    
    # Create progress bar for overall progress
    for file_idx, (full_path, relative_path) in enumerate(tqdm(audio_files, desc="Overall Progress"), 1):
        try:
            # Extract filename without extension
            filename = os.path.basename(relative_path)
            name, ext = os.path.splitext(filename)
            
            # Load audio
            audio, sr = librosa.load(full_path, sr=TARGET_SAMPLE_RATE)
            
            # Save clean version
            clean_output_path = os.path.join(CLEAN_OUTPUT_DIR, filename)
            sf.write(clean_output_path, audio, TARGET_SAMPLE_RATE, format='FLAC')
            
            # Generate and save noisy versions
            for noise_idx, noise_level in enumerate(NOISE_LEVELS, 1):
                noisy_audio = add_white_noise(audio, noise_level)
                
                # Create filename with noise level info
                noisy_filename = f"{name}_noise_level_{noise_idx}_{noise_level}.flac"
                noisy_output_path = os.path.join(NOISY_OUTPUT_DIR, noisy_filename)
                
                # Save noisy version
                sf.write(noisy_output_path, noisy_audio, TARGET_SAMPLE_RATE, format='FLAC')
            
            successful += 1
            
        except Exception as e:
            failed += 1
            print(f"\n❌ Error processing {relative_path}: {e}")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"Processing Complete!")
    print(f"{'='*80}\n")
    print(f"Total files processed: {total_files}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"\nOutput locations:")
    print(f"  Clean: {CLEAN_OUTPUT_DIR}")
    print(f"  Noisy: {NOISY_OUTPUT_DIR}")
    print(f"\nFiles per clean audio: 1 clean + {len(NOISE_LEVELS)} noisy = {1 + len(NOISE_LEVELS)} total")
    print(f"Total output files: {successful * (1 + len(NOISE_LEVELS))}")
    print(f"\n{'='*80}\n")


def preview_dataset():
    """
    Preview what will be processed without actually processing.
    Useful for checking before running the full process.
    """
    print(f"\n{'='*80}")
    print(f"Dataset Preview (No Processing)")
    print(f"{'='*80}\n")
    
    print(f"Scanning {LIBRISPEECH_ROOT}...")
    audio_files = find_all_audio_files(LIBRISPEECH_ROOT)
    
    if not audio_files:
        print(f"❌ No audio files found")
        return
    
    print(f"\n✅ Found {len(audio_files)} audio files")
    print(f"\nFirst 10 files:")
    for i, (full_path, relative_path) in enumerate(audio_files[:10], 1):
        print(f"  {i}. {relative_path}")
    
    if len(audio_files) > 10:
        print(f"  ... and {len(audio_files) - 10} more")
    
    print(f"\nWith {len(NOISE_LEVELS)} noise levels, this will create:")
    print(f"  • {len(audio_files)} clean files")
    print(f"  • {len(audio_files) * len(NOISE_LEVELS)} noisy files")
    print(f"  • {len(audio_files) * (1 + len(NOISE_LEVELS))} total files")
    
    # Estimate disk space (rough estimate)
    avg_size_mb = 0.5  # Average .flac file size in MB
    total_size_mb = len(audio_files) * (1 + len(NOISE_LEVELS)) * avg_size_mb
    print(f"\nEstimated disk space needed: ~{total_size_mb:.1f} MB (~{total_size_mb/1024:.2f} GB)")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'preview':
        # Preview mode - just show what would be processed
        preview_dataset()
    else:
        # Full processing mode
        print("\nTip: Run with 'preview' argument to see what will be processed first")
        print("Example: python process_all_librispeech.py preview\n")
        
        # Ask for confirmation
        response = input("This will process all audio files. Continue? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            process_all_audio_files()
        else:
            print("❌ Cancelled by user")
