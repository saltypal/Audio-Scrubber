import numpy as np
import os
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
# We will define paths locally if config is not available or for standalone usage
OUTPUT_DIR = Path(__file__).parent

def generate_ofdm_symbol(num_subcarriers, cp_len):
    """Generates a random OFDM symbol."""
    # 1. Generate random QPSK symbols (4-QAM)
    # 2 bits per symbol
    num_data_carriers = num_subcarriers  # Simplified: all carriers are data for now
    
    # Random bits mapped to QPSK: 1+1j, 1-1j, -1+1j, -1-1j
    bits = np.random.randint(0, 4, num_data_carriers)
    qam_map = np.array([1+1j, 1-1j, -1+1j, -1-1j])
    freq_domain_data = qam_map[bits]
    
    # 2. IFFT to move to Time Domain
    time_domain_data = np.fft.ifft(freq_domain_data) * np.sqrt(num_subcarriers) # Normalize energy
    
    # 3. Add Cyclic Prefix
    cp = time_domain_data[-cp_len:]
    ofdm_symbol = np.concatenate([cp, time_domain_data])
    
    return ofdm_symbol

def apply_channel_impairments(signal, snr_db, freq_offset_hz, sample_rate):
    """Adds noise, frequency offset, and multipath."""
    
    # 1. Multipath (Simple Taps)
    # Simulating a slight echo
    taps = np.array([1.0, 0.2+0.1j, 0.1-0.05j])
    signal_multipath = np.convolve(signal, taps, mode='same')
    
    # 2. Frequency Offset
    t = np.arange(len(signal)) / sample_rate
    phase_drift = np.exp(1j * 2 * np.pi * freq_offset_hz * t)
    signal_fo = signal_multipath * phase_drift
    
    # 3. AWGN Noise
    sig_power = np.mean(np.abs(signal_fo)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))) * np.sqrt(noise_power/2)
    
    return signal_fo + noise

def main():
    print("========================================")
    print("   OFDM DATA FACTORY (Python Edition)   ")
    print("========================================")
    
    # Parameters
    NUM_SAMPLES = 1_000_000  # Total samples to generate
    FFT_SIZE = 64
    CP_LEN = 16
    SAMPLE_RATE = 1_000_000 # 1 MHz
    SNR_DB = 15             # Moderate noise
    FREQ_OFFSET = 1000      # 1 kHz offset
    
    print(f"Generating {NUM_SAMPLES} samples...")
    print(f"FFT Size: {FFT_SIZE}, CP: {CP_LEN}")
    print(f"Target SNR: {SNR_DB} dB")
    
    clean_buffer = []
    noisy_buffer = []
    
    samples_generated = 0
    symbol_len = FFT_SIZE + CP_LEN
    
    while samples_generated < NUM_SAMPLES:
        # Generate one clean symbol
        clean_sym = generate_ofdm_symbol(FFT_SIZE, CP_LEN)
        
        # Apply channel model
        # We vary SNR slightly per symbol for robustness
        current_snr = np.random.uniform(SNR_DB - 5, SNR_DB + 5)
        current_fo = np.random.uniform(-FREQ_OFFSET, FREQ_OFFSET)
        
        noisy_sym = apply_channel_impairments(clean_sym, current_snr, current_fo, SAMPLE_RATE)
        
        clean_buffer.append(clean_sym)
        noisy_buffer.append(noisy_sym)
        
        samples_generated += len(clean_sym)
    
    # Flatten arrays
    clean_signal = np.concatenate(clean_buffer).astype(np.complex64)
    noisy_signal = np.concatenate(noisy_buffer).astype(np.complex64)
    
    # Trim to exact size
    clean_signal = clean_signal[:NUM_SAMPLES]
    noisy_signal = noisy_signal[:NUM_SAMPLES]
    
    # Save to files
    clean_path = OUTPUT_DIR / "clean_train.iq"
    noisy_path = OUTPUT_DIR / "noisy_train.iq"
    
    print(f"\nSaving to {OUTPUT_DIR}...")
    clean_signal.tofile(clean_path)
    noisy_signal.tofile(noisy_path)
    
    print(f"Saved {clean_path.name} ({clean_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Saved {noisy_path.name} ({noisy_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print("Done.")

if __name__ == "__main__":
    main()
