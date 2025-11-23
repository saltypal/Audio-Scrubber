
"""
Generate Synthetic OFDM Dataset using the Robust Library.
Use this to create new training data if needed.
"""
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ofdm.lib.transceiver import OFDMTransmitter
from src.ofdm.lib.utils import add_noise

def generate_dataset(output_dir="dataset/OFDM_NEW", num_samples=100000, snr_range=(0, 20)):
    print(f"Generating dataset in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    tx = OFDMTransmitter()
    
    # Generate random bytes
    # We want a continuous stream, but our transmitter works on packets.
    # Let's generate many small packets and concatenate.
    
    clean_signals = []
    noisy_signals = []
    
    total_samples = 0
    pbar = tqdm(total=num_samples)
    
    while total_samples < num_samples:
        # Random payload length (10 to 100 bytes)
        length = np.random.randint(10, 100)
        payload = np.random.bytes(length)
        
        # Transmit
        waveform, _ = tx.transmit(payload)
        
        # Add Noise (Random SNR)
        snr = np.random.uniform(snr_range[0], snr_range[1])
        noisy = add_noise(waveform, snr)
        
        clean_signals.append(waveform)
        noisy_signals.append(noisy)
        
        total_samples += len(waveform)
        pbar.update(len(waveform))
        
    pbar.close()
    
    # Concatenate
    full_clean = np.concatenate(clean_signals)[:num_samples]
    full_noisy = np.concatenate(noisy_signals)[:num_samples]
    
    # Save
    clean_path = os.path.join(output_dir, "clean_ofdm.iq")
    noisy_path = os.path.join(output_dir, "noisy_ofdm.iq")
    
    full_clean.tofile(clean_path)
    full_noisy.tofile(noisy_path)
    
    print(f"✅ Saved {num_samples} samples to {clean_path} and {noisy_path}")

if __name__ == "__main__":
    generate_dataset()
