import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import random

"""
Created by Satya with the help of Copilot

This script adds pre-generated noise (static.wav) to clean audio files
with specific SNR levels to create realistic noisy samples.
"""

CLEAN_AUDIO_DIR = r"dataset\instant\clean"
NOISE_FILE = r"dataset\noise\static.wav"  # Your pre-generated noise file
OUTPUT_DIR = r"dataset\instant\noisy"
TARGET_SNR_RANGE = (-10, 20)  # SNR range in dB (min, max)
NUM_NOISE_VERSIONS = 5  # Number of noisy versions per clean file


def add_noise_with_snr(clean_signal, noise_signal, target_snr_db):
    """
    Add noise to clean signal with a specific target SNR.
    
    Args:
        clean_signal: Clean audio numpy array
        noise_signal: Noise audio numpy array (same length as clean_signal)
        target_snr_db: Target Signal-to-Noise Ratio in dB
    
    Returns:
        noisy_signal: Mixed audio with specified SNR
        actual_snr_db: The actual SNR achieved
    """
    # Step 2: Measure the power of both signals
    power_signal = np.mean(clean_signal ** 2)
    power_noise_current = np.mean(noise_signal ** 2)
    
    # Step 3: Calculate the target noise power
    # Convert SNR from dB to linear scale
    target_snr_linear = 10 ** (target_snr_db / 10)
    
    # Find required noise power
    power_noise_target = power_signal / target_snr_linear
    
    # Step 4: Scale the noise
    # Find the power ratio
    power_ratio = power_noise_target / power_noise_current
    
    # Get the scale factor (square root because power is squared)
    scale_factor = np.sqrt(power_ratio)
    
    # Apply the scale factor to noise
    scaled_noise = noise_signal * scale_factor
    
    # Step 5: Create the final mix
    noisy_signal = clean_signal + scaled_noise
    
    # Clip to prevent distortion
    noisy_signal = np.clip(noisy_signal, -1.0, 1.0)
    
    # Calculate actual SNR for verification
    power_noise_final = np.mean(scaled_noise ** 2)
    actual_snr_linear = power_signal / power_noise_final
    actual_snr_db = 10 * np.log10(actual_snr_linear)
    
    return noisy_signal, actual_snr_db


def get_noise_chunk(noise_audio, noise_sr, target_length, target_sr):
    """
    Extract a random chunk of noise with the target length.
    
    Args:
        noise_audio: Full noise audio array
        noise_sr: Sample rate of noise audio
        target_length: Desired length in samples
        target_sr: Target sample rate
    
    Returns:
        noise_chunk: Random chunk of noise with target length
    """
    # Resample noise if sample rates don't match
    if noise_sr != target_sr:
        noise_audio = librosa.resample(noise_audio, orig_sr=noise_sr, target_sr=target_sr)
    
    # If noise is shorter than target, loop it
    if len(noise_audio) < target_length:
        repeats = (target_length // len(noise_audio)) + 1
        noise_audio = np.tile(noise_audio, repeats)
    
    # Extract random chunk
    if len(noise_audio) > target_length:
        start_idx = random.randint(0, len(noise_audio) - target_length)
        noise_chunk = noise_audio[start_idx:start_idx + target_length]
    else:
        noise_chunk = noise_audio[:target_length]
    
    return noise_chunk


def process_audio_files():
    """
    Load clean audio files, add pre-generated noise with specific SNR levels,
    and save them as FLAC.
    """
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load the noise file once
    print(f"Loading noise file: {NOISE_FILE}")
    noise_audio, noise_sr = librosa.load(NOISE_FILE, sr=None)
    print(f"Noise file loaded: {len(noise_audio)} samples at {noise_sr} Hz\n")
    
    # Get all clean audio files
    audio_files = [f for f in os.listdir(CLEAN_AUDIO_DIR) if f.endswith('.flac')]
    
    if not audio_files:
        print(f"No audio files found in {CLEAN_AUDIO_DIR}")
        return
    
    print(f"Found {len(audio_files)} clean audio files")
    print(f"Will create {NUM_NOISE_VERSIONS} noisy versions per file")
    print(f"SNR range: {TARGET_SNR_RANGE[0]} dB to {TARGET_SNR_RANGE[1]} dB\n")
    
    for file_idx, filename in enumerate(audio_files, 1):
        filepath = os.path.join(CLEAN_AUDIO_DIR, filename)
        
        try:
            # Load clean audio
            clean_audio, sr = librosa.load(filepath, sr=None)
            print(f"[{file_idx}/{len(audio_files)}] Processing: {filename}")
            
            # Create multiple noisy versions with different SNRs
            for version in range(NUM_NOISE_VERSIONS):
                # Step 1: Get ingredients
                target_snr_db = random.uniform(TARGET_SNR_RANGE[0], TARGET_SNR_RANGE[1])
                noise_chunk = get_noise_chunk(noise_audio, noise_sr, len(clean_audio), sr)
                
                # Add noise with specific SNR
                noisy_audio, actual_snr_db = add_noise_with_snr(clean_audio, noise_chunk, target_snr_db)
                
                # Create output filename
                name, ext = os.path.splitext(filename)
                noisy_filename = f"{name}_snr_{target_snr_db:.1f}dB.flac"
                output_path = os.path.join(OUTPUT_DIR, noisy_filename)
                
                # Save noisy audio
                sf.write(output_path, noisy_audio, sr, format='FLAC')
                print(f"  └─ Saved: {noisy_filename} (Target: {target_snr_db:.1f} dB, Actual: {actual_snr_db:.1f} dB)")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"\nCompleted! Noisy audio files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    process_audio_files()