import os
import sys
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import Paths, AudioSettings, NoiseSettings

"""
Advanced LibriSpeech Dataset Processor

This script processes the entire LibriSpeech dataset with flexible options:
- Process single or multiple subsets (dev-clean, train-clean-100, etc.)
- Add noise at configurable levels
- Maintain or flatten directory structure
- Preview before processing
- Resume interrupted processing

Usage:
    # Preview what will be processed
    python process_librispeech_advanced.py --preview
    
    # Process dev-clean subset
    python process_librispeech_advanced.py --subset dev-clean
    
    # Process multiple subsets
    python process_librispeech_advanced.py --subset dev-clean train-clean-100
    
    # Process all available subsets
    python process_librispeech_advanced.py --all
    
    # Custom noise levels
    python process_librispeech_advanced.py --subset dev-clean --noise 0.01 0.05 0.1
"""

# Import from central config
LIBRISPEECH_ROOT = Paths.LIBRISPEECH_ROOT
OUTPUT_ROOT = Paths.LIBRISPEECH_PROCESSED

# Import settings from central config
DEFAULT_NOISE_LEVELS = NoiseSettings.NOISE_LEVELS
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


def find_all_audio_files(subset_path):
    """
    Recursively find all .flac files in a LibriSpeech subset.
    
    Args:
        subset_path: Path to LibriSpeech subset directory
    
    Returns:
        List of full paths to .flac files
    """
    audio_files = []
    
    for root, dirs, files in os.walk(subset_path):
        for file in files:
            if file.endswith('.flac'):
                full_path = os.path.join(root, file)
                audio_files.append(full_path)
    
    return audio_files


def get_available_subsets():
    """
    Scan LibriSpeech root directory and find available subsets.
    
    Returns:
        List of subset names (e.g., ['dev-clean', 'train-clean-100'])
    """
    if not os.path.exists(LIBRISPEECH_ROOT):
        return []
    
    subsets = []
    for item in os.listdir(LIBRISPEECH_ROOT):
        item_path = os.path.join(LIBRISPEECH_ROOT, item)
        if os.path.isdir(item_path):
            # Check if it looks like a LibriSpeech subset (contains speaker dirs)
            has_audio = False
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path) and subitem.isdigit():
                    has_audio = True
                    break
            if has_audio:
                subsets.append(item)
    
    return sorted(subsets)


def process_subset(subset_name, noise_levels, skip_existing=True):
    """
    Process a single LibriSpeech subset.
    
    Args:
        subset_name: Name of the subset (e.g., 'dev-clean')
        noise_levels: List of noise levels to apply
        skip_existing: If True, skip files that already exist
    
    Returns:
        Tuple of (successful_count, failed_count)
    """
    subset_path = os.path.join(LIBRISPEECH_ROOT, subset_name)
    clean_output_dir = os.path.join(OUTPUT_ROOT, subset_name, "clean")
    noisy_output_dir = os.path.join(OUTPUT_ROOT, subset_name, "noisy")
    
    # Create output directories
    Path(clean_output_dir).mkdir(parents=True, exist_ok=True)
    Path(noisy_output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Processing: {subset_name}")
    print(f"{'='*80}")
    print(f"Source: {subset_path}")
    print(f"Clean output: {clean_output_dir}")
    print(f"Noisy output: {noisy_output_dir}\n")
    
    # Find all audio files
    print("Scanning for audio files...")
    audio_files = find_all_audio_files(subset_path)
    
    if not audio_files:
        print(f"❌ No audio files found in {subset_name}")
        return 0, 0
    
    print(f"✅ Found {len(audio_files)} audio files\n")
    
    # Process each file
    successful = 0
    failed = 0
    skipped = 0
    
    for full_path in tqdm(audio_files, desc=f"Processing {subset_name}"):
        try:
            filename = os.path.basename(full_path)
            name, ext = os.path.splitext(filename)
            
            clean_output_path = os.path.join(clean_output_dir, filename)
            
            # Skip if already processed and skip_existing is True
            if skip_existing and os.path.exists(clean_output_path):
                # Check if all noisy versions exist too
                all_exist = True
                for noise_idx, noise_level in enumerate(noise_levels, 1):
                    noisy_filename = f"{name}_noise_level_{noise_idx}_{noise_level}.flac"
                    noisy_output_path = os.path.join(noisy_output_dir, noisy_filename)
                    if not os.path.exists(noisy_output_path):
                        all_exist = False
                        break
                
                if all_exist:
                    skipped += 1
                    continue
            
            # Load audio
            audio, sr = librosa.load(full_path, sr=TARGET_SAMPLE_RATE)
            
            # Save clean version
            sf.write(clean_output_path, audio, TARGET_SAMPLE_RATE, format='FLAC')
            
            # Generate and save noisy versions
            for noise_idx, noise_level in enumerate(noise_levels, 1):
                noisy_audio = add_white_noise(audio, noise_level)
                noisy_filename = f"{name}_noise_level_{noise_idx}_{noise_level}.flac"
                noisy_output_path = os.path.join(noisy_output_dir, noisy_filename)
                sf.write(noisy_output_path, noisy_audio, TARGET_SAMPLE_RATE, format='FLAC')
            
            successful += 1
            
        except Exception as e:
            failed += 1
            tqdm.write(f"❌ Error processing {os.path.basename(full_path)}: {e}")
    
    # Summary for this subset
    print(f"\n{'='*80}")
    print(f"Subset '{subset_name}' Complete")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏭ Skipped: {skipped}")
    print(f"{'='*80}\n")
    
    return successful, failed


def preview_processing(subsets, noise_levels):
    """
    Preview what will be processed without actually processing.
    
    Args:
        subsets: List of subset names to preview
        noise_levels: List of noise levels
    """
    print(f"\n{'='*80}")
    print(f"Processing Preview")
    print(f"{'='*80}\n")
    
    total_files = 0
    subset_details = []
    
    for subset_name in subsets:
        subset_path = os.path.join(LIBRISPEECH_ROOT, subset_name)
        
        if not os.path.exists(subset_path):
            print(f"⚠ Subset '{subset_name}' not found at {subset_path}")
            continue
        
        audio_files = find_all_audio_files(subset_path)
        total_files += len(audio_files)
        subset_details.append((subset_name, len(audio_files)))
        
        print(f"📁 {subset_name}: {len(audio_files)} audio files")
    
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    print(f"Total subsets: {len(subset_details)}")
    print(f"Total audio files: {total_files}")
    print(f"Noise levels: {len(noise_levels)} ({noise_levels})")
    print(f"\nOutput files:")
    print(f"  • {total_files} clean files")
    print(f"  • {total_files * len(noise_levels)} noisy files")
    print(f"  • {total_files * (1 + len(noise_levels))} total files")
    
    # Estimate disk space
    avg_size_mb = 0.5  # Conservative estimate
    total_size_mb = total_files * (1 + len(noise_levels)) * avg_size_mb
    print(f"\nEstimated disk space: ~{total_size_mb:.1f} MB (~{total_size_mb/1024:.2f} GB)")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Process LibriSpeech dataset and add noise',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview processing
  python %(prog)s --preview
  
  # Process dev-clean
  python %(prog)s --subset dev-clean
  
  # Process multiple subsets
  python %(prog)s --subset dev-clean train-clean-100
  
  # Process all available subsets
  python %(prog)s --all
  
  # Custom noise levels
  python %(prog)s --subset dev-clean --noise 0.01 0.05 0.1
        """
    )
    
    parser.add_argument('--subset', nargs='+', help='Subset(s) to process (e.g., dev-clean)')
    parser.add_argument('--all', action='store_true', help='Process all available subsets')
    parser.add_argument('--noise', nargs='+', type=float, help='Noise levels to apply')
    parser.add_argument('--preview', action='store_true', help='Preview without processing')
    parser.add_argument('--no-skip', action='store_true', help='Reprocess existing files')
    
    args = parser.parse_args()
    
    # Determine noise levels
    noise_levels = args.noise if args.noise else DEFAULT_NOISE_LEVELS
    
    # Determine which subsets to process
    if args.all:
        subsets = get_available_subsets()
        if not subsets:
            print(f"❌ No LibriSpeech subsets found in {LIBRISPEECH_ROOT}")
            return
        print(f"Found subsets: {', '.join(subsets)}\n")
    elif args.subset:
        subsets = args.subset
    else:
        # Show available subsets and prompt
        available = get_available_subsets()
        if not available:
            print(f"❌ No LibriSpeech subsets found in {LIBRISPEECH_ROOT}")
            return
        
        print(f"Available subsets in {LIBRISPEECH_ROOT}:")
        for i, subset in enumerate(available, 1):
            print(f"  {i}. {subset}")
        print("\nPlease specify --subset or --all")
        return
    
    # Preview mode
    if args.preview:
        preview_processing(subsets, noise_levels)
        return
    
    # Confirm before processing
    print(f"\n{'='*80}")
    print(f"Ready to Process")
    print(f"{'='*80}")
    print(f"Subsets: {', '.join(subsets)}")
    print(f"Noise levels: {noise_levels}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Sample rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"Skip existing: {not args.no_skip}")
    print(f"{'='*80}\n")
    
    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled by user")
        return
    
    # Process each subset
    total_successful = 0
    total_failed = 0
    
    for subset_name in subsets:
        successful, failed = process_subset(subset_name, noise_levels, skip_existing=not args.no_skip)
        total_successful += successful
        total_failed += failed
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"All Processing Complete!")
    print(f"{'='*80}")
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    print(f"Output location: {OUTPUT_ROOT}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
