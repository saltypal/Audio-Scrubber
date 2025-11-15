import numpy as np
import soundfile as sf
from pathlib import Path

"""
Created by Satya with the help of Copilot

Generates a white noise file for audio denoising experiments.
"""

# --- Config ---
SAMPLING_RATE = 22050  # Standard rate for ML projects
DURATION_SECONDS = 600  # 10 minutes
OUTPUT_DIR = r"dataset\noise"
NOISE_FILE_NAME = "static.flac"
# ---

print("Generating white noise...")

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Calculate total number of samples
num_samples = SAMPLING_RATE * DURATION_SECONDS

# Generate the noise
# np.random.normal(mean, std_dev, size)
# This is your "Additive White Gaussian Noise" (AWGN)
noise = np.random.normal(0, 1, num_samples)

# "Normalize" the noise so it's not too loud (prevents clipping)
noise = noise / np.max(np.abs(noise))

# Save the noise to a FLAC file
output_path = f"{OUTPUT_DIR}\\{NOISE_FILE_NAME}"
try:
    sf.write(output_path, noise, SAMPLING_RATE, format='FLAC')
    print(f"\nSuccess! 🚀")
    print(f"Created a {DURATION_SECONDS // 60}-minute noise file: {output_path}")
except Exception as e:
    print(f"\nError writing file: {e}")
