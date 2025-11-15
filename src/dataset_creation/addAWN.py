import os 
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

"""
Created by Satya At 2:23PM with the help of copilot

This script takes all .flac audio files from instant\clean, 
adds white noise at several levels to each, and saves the noisy versions as .flac files in dataset\noisy.

adding noise to a signal using pthon
https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python

"""

# CLEAN_AUDIO_DIR = r"dataset\instant\clean"
# OUTPUT_DIR = r"dataset\instant\noisy"


CLEAN_AUDIO_DIR = r"testing"
OUTPUT_DIR = r"testing"

print("We will add noise with Signal To Noise Ratio levels. ")
NOISE_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.1]



def add_white_noise(audio, noise_level):
    """
    Add white noise to audio signal.
    Goal: We have to create noisy audio from 
    """
    noise = np.random.normal(0, noise_level, len(audio)) # this makes the noise equal to the lenght of the audio file 
    noisy_audio = audio + noise # this adds the noise to the audio file
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0) # this clips the audio, like a double ended clipper
    return noisy_audio



def process_audio_files():
    """
    Load clean audio files, add white noise at different levels, and save them as FLAC.
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    audio_files = [f for f in os.listdir(CLEAN_AUDIO_DIR) if f.endswith('.flac')]
    if not audio_files:
        print(f"No audio files found in {CLEAN_AUDIO_DIR}")
        return
    print(f"Found {len(audio_files)} audio files")
    print(f"Processing with {len(NOISE_LEVELS)} noise levels: {NOISE_LEVELS}\n")
    for file_idx, filename in enumerate(audio_files, 1):
        filepath = os.path.join(CLEAN_AUDIO_DIR, filename)
        try:
            audio, sr = librosa.load(filepath, sr=None)
            print(f"[{file_idx}/{len(audio_files)}] Loading: {filename}")
            for noise_idx, noise_level in enumerate(NOISE_LEVELS, 1):
                noisy_audio = add_white_noise(audio, noise_level)
                name, ext = os.path.splitext(filename)
                noisy_filename = f"{name}_noise_level_{noise_idx}_{noise_level}.flac"
                output_path = os.path.join(OUTPUT_DIR, noisy_filename)
                sf.write(output_path, noisy_audio, sr, format='FLAC')
                print(f"  └─ Saved: {noisy_filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    print(f"\nCompleted! Noisy audio files saved to {OUTPUT_DIR}")



if __name__ == "__main__":
    process_audio_files()