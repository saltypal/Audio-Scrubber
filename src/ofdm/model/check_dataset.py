
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def check_iq_file(filepath):
    path = Path(filepath)
    print(f"Checking file: {path}")
    
    if not path.exists():
        print("❌ File does not exist!")
        return

    file_size = path.stat().st_size
    print(f"📁 File size: {file_size / 1024 / 1024:.2f} MB")
    
    if file_size == 0:
        print("❌ File is empty!")
        return

    # Load data (GNU Radio usually saves as complex64)
    try:
        data = np.fromfile(path, dtype=np.complex64)
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return

    num_samples = len(data)
    print(f"🔢 Number of samples: {num_samples}")
    
    if num_samples == 0:
        print("❌ No samples found (but file size > 0?)")
        return

    # Check for NaNs or Infs
    if np.isnan(data).any():
        print("⚠️  Warning: Data contains NaNs!")
    if np.isinf(data).any():
        print("⚠️  Warning: Data contains Infs!")

    # Statistics
    real_part = data.real
    imag_part = data.imag
    
    print("\n📊 Statistics:")
    print(f"   Real: min={real_part.min():.4f}, max={real_part.max():.4f}, mean={real_part.mean():.4f}")
    print(f"   Imag: min={imag_part.min():.4f}, max={imag_part.max():.4f}, mean={imag_part.mean():.4f}")
    
    power = np.mean(np.abs(data)**2)
    print(f"   Average Power: {power:.4f}")
    
    if power == 0:
        print("❌ Signal power is ZERO. The file contains silence/zeros.")
    elif power < 1e-6:
        print("⚠️  Signal power is extremely low. Might be empty noise.")
    else:
        print("✅ Signal power looks reasonable.")

    # Visualization
    print("\n📈 Generating plots...")
    
    output_dir = Path("output/checks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Time Domain (First 1000 samples)
    limit = min(1000, num_samples)
    plt.subplot(2, 2, 1)
    plt.plot(data[:limit].real, label='Real')
    plt.plot(data[:limit].imag, label='Imag', alpha=0.7)
    plt.title(f"Time Domain (First {limit} samples)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Constellation (First 5000 samples)
    limit_const = min(5000, num_samples)
    plt.subplot(2, 2, 2)
    plt.scatter(data[:limit_const].real, data[:limit_const].imag, alpha=0.1, s=1)
    plt.title(f"Constellation (First {limit_const} samples)")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Histogram
    plt.subplot(2, 2, 3)
    plt.hist(data[:10000].real, bins=50, alpha=0.5, label='Real')
    plt.hist(data[:10000].imag, bins=50, alpha=0.5, label='Imag')
    plt.title("Histogram (Distribution)")
    plt.legend()
    
    # PSD (Power Spectral Density)
    plt.subplot(2, 2, 4)
    plt.psd(data[:min(10000, num_samples)], NFFT=1024, Fs=1.0)
    plt.title("Power Spectral Density")
    
    plt.tight_layout()
    plot_path = output_dir / f"check_{path.stem}.png"
    plt.savefig(plot_path)
    print(f"✅ Plot saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Check both clean and noisy if they exist
    base_path = Path("dataset/OFDM")
    
    clean_path = base_path / "clean_ofdm.iq"
    if clean_path.exists():
        check_iq_file(clean_path)
    else:
        print(f"File not found: {clean_path}")
        
    print("-" * 50)
    
    noisy_path = base_path / "noisy_ofdm.iq"
    if noisy_path.exists():
        check_iq_file(noisy_path)
    else:
        print(f"File not found: {noisy_path}")
