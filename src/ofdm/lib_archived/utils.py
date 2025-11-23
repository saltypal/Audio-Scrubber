
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_comparison(clean, noisy, denoised, save_path="comparison.png"):
    """Visualize the three stages of the signal."""
    plt.figure(figsize=(15, 10))
    
    # 1. Time Domain
    plt.subplot(3, 1, 1)
    plt.plot(clean[:200].real, label='Clean', alpha=0.7)
    plt.plot(noisy[:200].real, label='Noisy', alpha=0.7)
    plt.plot(denoised[:200].real, label='Denoised', alpha=0.7)
    plt.title("Time Domain (Real Part)")
    plt.legend()
    
    # 2. Spectrum
    plt.subplot(3, 1, 2)
    plt.psd(clean[:1024], Fs=1.0, label='Clean')
    plt.psd(noisy[:1024], Fs=1.0, label='Noisy')
    plt.psd(denoised[:1024], Fs=1.0, label='Denoised')
    plt.title("Power Spectral Density")
    plt.legend()
    
    # 3. Constellation (After Demodulation simulation)
    # We just plot the raw samples as a cloud
    plt.subplot(3, 1, 3)
    plt.scatter(noisy[:1000].real, noisy[:1000].imag, alpha=0.3, label='Noisy', s=1)
    plt.scatter(denoised[:1000].real, denoised[:1000].imag, alpha=0.3, label='Denoised', s=1)
    plt.title("Signal Constellation (Time Domain Cloud)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def add_noise(signal, snr_db):
    """Add AWGN noise."""
    power = np.mean(np.abs(signal)**2)
    noise_power = power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise
