"""
Advanced White Noise Addition with SNR Control (AWN2.py)

This script adds noise to audio files using Signal-to-Noise Ratio (SNR) in dB.
This is the industry-standard approach for noise addition.

SNR = 10 * log10(signal_power / noise_power)

Features:
- SNR-based noise addition (engineering standard)
- Configurable target SNR levels in dB
- Works with any audio dataset (LibriSpeech, Music, etc.)
- Preserves original audio quality
- Creates matched clean/noisy pairs

Usage:
    python src/dataset_creation/AWN2.py --input dataset/LibriSpeech/dev-clean \
                                         --output dataset/LibriSpeech_processed_snr \
                                         --snr_levels 0 5 10 15 20
    
    # For music
    python src/dataset_creation/AWN2.py --input dataset/music/raw \
                                         --output dataset/music_processed \
                                         --snr_levels 0 5 10 15 20

Created by Satya with Copilot @ 15/11/25
"""

import os
import sys
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import AudioSettings


def calculate_rms(audio):
    """
    Calculate Root Mean Square (RMS) of audio signal.
    RMS is used to measure the power of the signal.
    
    Args:
        audio: Audio signal array
    
    Returns:
        RMS value (scalar)
    """
    return np.sqrt(np.mean(audio ** 2))


def calculate_signal_power(audio):
    """
    Calculate signal power.
    Power = RMS^2
    
    Args:
        audio: Audio signal array
    
    Returns:
        Signal power (scalar)
    """
    rms = calculate_rms(audio)
    return rms ** 2


def add_noise_snr(clean_audio, target_snr_db):
    """
    Add white Gaussian noise to achieve target SNR in dB.
    
    SNR (dB) = 10 * log10(signal_power / noise_power)
    
    Algorithm:
    1. Calculate signal power
    2. Calculate required noise power for target SNR
    3. Generate white noise with calculated power
    4. Add noise to signal
    
    Args:
        clean_audio: Clean audio signal (numpy array)
        target_snr_db: Target SNR in decibels (0, 5, 10, 15, 20 dB)
    
    Returns:
        noisy_audio: Audio with added noise
        actual_snr: Actual achieved SNR (for verification)
    """
    # Calculate signal power
    signal_power = calculate_signal_power(clean_audio)
    
    # Convert target SNR from dB to linear scale
    # SNR_linear = 10^(SNR_dB / 10)
    snr_linear = 10 ** (target_snr_db / 10)
    
    # Calculate required noise power
    # noise_power = signal_power / SNR_linear
    noise_power = signal_power / snr_linear
    
    # Generate white Gaussian noise with calculated power
    # std_dev = sqrt(noise_power)
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, len(clean_audio))
    
    # Add noise to signal
    noisy_audio = clean_audio + noise
    
    # Verify actual SNR (for debugging/validation)
    actual_noise_power = calculate_signal_power(noise)
    actual_snr_db = 10 * np.log10(signal_power / actual_noise_power) if actual_noise_power > 0 else float('inf')
    
    return noisy_audio, actual_snr_db


def find_all_audio_files(directory, extensions=['.flac', '.wav', '.mp3']):
    """
    Recursively find all audio files in directory.
    
    Args:
        directory: Root directory to search
        extensions: List of audio file extensions to find
    
    Returns:
        List of audio file paths (relative to directory)
    """
    audio_files = []
    directory = Path(directory)
    
    for ext in extensions:
        audio_files.extend(directory.rglob(f'*{ext}'))
    
    return sorted(audio_files)


def process_audio_file(input_path, clean_output_dir, noisy_output_dir, snr_levels, sample_rate):
    """
    Process a single audio file: copy clean version and create noisy versions.
    
    Args:
        input_path: Path to input audio file
        clean_output_dir: Directory for clean audio
        noisy_output_dir: Directory for noisy audio
        snr_levels: List of SNR levels in dB
        sample_rate: Target sample rate
    
    Returns:
        Number of files created (1 clean + N noisy)
    """
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=sample_rate)
        
        # Get relative path structure
        input_path = Path(input_path)
        # Find base directory by going up until we find a parent that exists in the path
        # This preserves the directory structure from the input
        try:
            # Try to preserve full structure from input dir
            base_dir = input_path.parent
            while base_dir.parent != base_dir:  # Stop at root
                if 'LibriSpeech' in str(base_dir) or 'music' in str(base_dir):
                    break
                base_dir = base_dir.parent
            relative_path = input_path.relative_to(base_dir.parent) if base_dir.parent != base_dir else input_path
        except (ValueError, IndexError):
            # Fallback: just use filename
            relative_path = input_path
        
        # Create output paths
        clean_output_path = Path(clean_output_dir) / relative_path
        clean_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save clean audio
        sf.write(clean_output_path, audio, sample_rate, format='FLAC')
        files_created = 1
        
        # Create noisy versions for each SNR level
        for snr_db in snr_levels:
            # Add noise
            noisy_audio, actual_snr = add_noise_snr(audio, snr_db)
            
            # Create noisy filename: original_name_snr_10dB.flac
            noisy_filename = f"{input_path.stem}_snr_{snr_db}dB{input_path.suffix}"
            noisy_output_path = Path(noisy_output_dir) / relative_path.parent / noisy_filename
            noisy_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save noisy audio
            sf.write(noisy_output_path, noisy_audio, sample_rate, format='FLAC')
            files_created += 1
        
        return files_created
    
    except Exception as e:
        print(f"\n❌ Error processing {input_path}: {e}")
        return 0


def process_dataset(input_dir, output_dir, snr_levels, sample_rate=22050, preview=False):
    """
    Process entire dataset: create clean and noisy versions.
    
    Args:
        input_dir: Input directory with raw audio
        output_dir: Output directory for processed audio
        snr_levels: List of SNR levels in dB (e.g., [0, 5, 10, 15, 20])
        sample_rate: Target sample rate
        preview: If True, only show what would be processed
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    clean_output_dir = output_dir / "clean"
    noisy_output_dir = output_dir / "noisy"
    
    if not preview:
        clean_output_dir.mkdir(parents=True, exist_ok=True)
        noisy_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    print(f"\n🔍 Scanning for audio files in: {input_dir}")
    audio_files = find_all_audio_files(input_dir)
    
    if not audio_files:
        print(f"❌ No audio files found in {input_dir}")
        return
    
    print(f"✅ Found {len(audio_files)} audio files")
    print(f"\n📊 Processing Configuration:")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Clean dir: {clean_output_dir}")
    print(f"   Noisy dir: {noisy_output_dir}")
    print(f"   SNR levels: {snr_levels} dB")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Files per input: 1 clean + {len(snr_levels)} noisy = {1 + len(snr_levels)} total")
    print(f"   Expected output: {len(audio_files)} × {1 + len(snr_levels)} = {len(audio_files) * (1 + len(snr_levels))} files")
    
    if preview:
        print(f"\n👁️  PREVIEW MODE - No files will be created")
        print(f"\nSample files that would be processed:")
        for i, file in enumerate(audio_files[:5], 1):
            print(f"   {i}. {file.relative_to(input_dir)}")
        if len(audio_files) > 5:
            print(f"   ... and {len(audio_files) - 5} more")
        return
    
    # Process all files
    print(f"\n🚀 Processing audio files...\n")
    total_files_created = 0
    
    progress_bar = tqdm(audio_files, desc="Processing")
    for audio_file in progress_bar:
        files_created = process_audio_file(
            audio_file,
            clean_output_dir,
            noisy_output_dir,
            snr_levels,
            sample_rate
        )
        total_files_created += files_created
        progress_bar.set_postfix({
            'created': total_files_created,
            'file': audio_file.name[:30]
        })
    
    print(f"\n{'='*60}")
    print(f"✅ Processing Complete!")
    print(f"{'='*60}")
    print(f"Input files: {len(audio_files)}")
    print(f"Output files: {total_files_created}")
    print(f"Clean files: {len(audio_files)}")
    print(f"Noisy files: {len(audio_files) * len(snr_levels)}")
    print(f"\nOutput locations:")
    print(f"  Clean: {clean_output_dir}")
    print(f"  Noisy: {noisy_output_dir}")
    print(f"{'='*60}\n")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='Add SNR-controlled noise to audio dataset (AWN2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process LibriSpeech dev-clean with 5 SNR levels
  python AWN2.py --input dataset/LibriSpeech/dev-clean --output dataset/LibriSpeech_processed_snr

  # Process music dataset
  python AWN2.py --input dataset/music/raw --output dataset/music_processed --snr_levels 0 5 10 15 20

  # Preview without processing
  python AWN2.py --input dataset/LibriSpeech/dev-clean --output dataset/LibriSpeech_processed_snr --preview

  # Custom sample rate for music (44.1kHz)
  python AWN2.py --input dataset/music --output dataset/music_processed --sample_rate 44100
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with audio files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for processed files'
    )
    
    parser.add_argument(
        '--snr_levels',
        type=int,
        nargs='+',
        default=[0, 5, 10, 15, 20],
        help='SNR levels in dB (default: 0 5 10 15 20)'
    )
    
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=AudioSettings.SAMPLE_RATE,
        help=f'Target sample rate in Hz (default: {AudioSettings.SAMPLE_RATE})'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview mode - show what would be processed without creating files'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input):
        print(f"❌ Error: Input directory does not exist: {args.input}")
        return
    
    # Process dataset
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        snr_levels=args.snr_levels,
        sample_rate=args.sample_rate,
        preview=args.preview
    )


if __name__ == "__main__":
    main()
