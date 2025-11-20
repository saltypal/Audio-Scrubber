"""
Quick script to inspect OFDM dataset files.
"""
import numpy as np
from pathlib import Path

def inspect_dataset():
    dataset_dir = Path("dataset/OFDM")
    clean_path = dataset_dir / "clean_ofdm.iq"
    noisy_path = dataset_dir / "noisy_ofdm.iq"
    
    print("=" * 60)
    print("         OFDM DATASET INSPECTOR")
    print("=" * 60)
    
    if not clean_path.exists():
        print(f"❌ {clean_path} not found!")
        print("\n💡 Generate dataset first:")
        print("   python dataset_ofdm/ofdm_dataset_creation.py")
        return
    
    # Load files
    clean = np.fromfile(clean_path, dtype=np.complex64)
    noisy = np.fromfile(noisy_path, dtype=np.complex64)
    
    print(f"\n📁 Clean: {clean_path}")
    print(f"   Samples: {len(clean):,}")
    print(f"   Size: {clean_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Dtype: {clean.dtype}")
    print(f"   Range: [{clean.real.min():.3f}, {clean.real.max():.3f}]")
    
    print(f"\n📁 Noisy: {noisy_path}")
    print(f"   Samples: {len(noisy):,}")
    print(f"   Size: {noisy_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Dtype: {noisy.dtype}")
    print(f"   Range: [{noisy.real.min():.3f}, {noisy.real.max():.3f}]")
    
    # Calculate SNR
    noise = noisy - clean
    signal_power = np.mean(np.abs(clean) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    print(f"\n📊 Statistics:")
    print(f"   Signal Power: {signal_power:.6f}")
    print(f"   Noise Power: {noise_power:.6f}")
    print(f"   Estimated SNR: {snr_db:.2f} dB")
    
    # Training info
    chunk_size = 1024
    num_chunks = len(clean) // chunk_size
    print(f"\n🎯 Training Info (chunk_size=1024):")
    print(f"   Total chunks: {num_chunks:,}")
    print(f"   At batch_size=32: {num_chunks // 32:,} batches per epoch")
    
    print("\n✅ Dataset ready for training!")
    print("\n🚀 To train:")
    print("   python src/ofdm/model/train_ofdm.py --epochs 50 --batch_size 32")
    print("=" * 60)

if __name__ == "__main__":
    inspect_dataset()
