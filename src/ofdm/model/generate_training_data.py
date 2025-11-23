"""
================================================================================
OFDM TRAINING DATA GENERATOR
================================================================================
Generates clean and noisy OFDM waveform pairs for AI model training.
Uses the new OFDM pipeline implementation to ensure compatibility.

Output:
- dataset/OFDM/clean_ofdm_new.iq
- dataset/OFDM/noisy_ofdm_new.iq

Usage:
    python src/ofdm/model/generate_training_data.py
    python src/ofdm/model/generate_training_data.py --samples 200000000 --snr-range 0 25
================================================================================
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ofdm.core import OFDMTransceiver, add_awgn_noise


def generate_diverse_texts(num_texts=1000):
    """
    Generate diverse text samples for training.
    Uses various sources to ensure model generalizes well.
    """
    texts = []
    
    # 1. Common phrases
    common_phrases = [
        "The quick brown fox jumps over the lazy dog",
        "Hello World! This is a test message",
        "Lorem ipsum dolor sit amet consectetur adipiscing elit",
        "OFDM transmission test with AI denoising",
        "Wireless communication system verification",
        "Digital signal processing pipeline active",
        "Software defined radio experiment running",
        "Real-time data transmission in progress",
    ]
    
    # 2. Numbers and special characters
    for i in range(100):
        texts.append(f"Data packet {i:04d} - Status: OK - Checksum: {i*123:08x}")
        texts.append(f"Timestamp: {i}ms | Signal: {i%10}/10 | Error: {i%5}")
    
    # 3. Random ASCII
    for _ in range(200):
        length = np.random.randint(20, 200)
        random_text = ''.join([chr(np.random.randint(32, 127)) for _ in range(length)])
        texts.append(random_text)
    
    # 4. Mixed case and punctuation
    punctuation_texts = [
        "ERROR! System malfunction detected!!!",
        "SUCCESS: All tests passed. Ready for deployment.",
        "WARNING: Signal strength below threshold (SNR < 10dB)",
        "INFO: Processing 1,234,567 samples at 2.0 MSPS",
    ]
    
    # 5. Repeat common phrases to reach target
    all_texts = common_phrases * 20 + texts + punctuation_texts * 10
    
    # Shuffle
    np.random.shuffle(all_texts)
    
    return all_texts[:num_texts]


def generate_dataset(
    target_samples=100_000_000,
    snr_range=(0, 25),
    output_dir='dataset/OFDM',
    chunk_save_interval=10_000_000
):
    """
    Generate training dataset for OFDM AI denoising.
    
    Args:
        target_samples: Number of IQ samples to generate (default: 100M)
        snr_range: Tuple of (min_snr, max_snr) in dB
        output_dir: Directory to save output files
        chunk_save_interval: Save to disk every N samples (to avoid RAM overflow)
    """
    print("="*80)
    print(" "*20 + "OFDM TRAINING DATA GENERATOR")
    print("="*80)
    print(f"\nTarget Samples: {target_samples:,}")
    print(f"SNR Range: {snr_range[0]} to {snr_range[1]} dB")
    print(f"Output Directory: {output_dir}")
    print(f"Chunk Save Interval: {chunk_save_interval:,} samples")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output files
    clean_file = output_dir / 'clean_ofdm_new.iq'
    noisy_file = output_dir / 'noisy_ofdm_new.iq'
    
    # Initialize transceiver
    transceiver = OFDMTransceiver()
    
    # Generate diverse texts
    print("\n📝 Generating diverse text samples...")
    texts = generate_diverse_texts(num_texts=2000)
    print(f"   Generated {len(texts)} unique text samples")
    
    # SNR levels (uniform distribution)
    snr_levels = np.linspace(snr_range[0], snr_range[1], 10)
    print(f"\n🎚️  SNR Levels: {snr_levels}")
    
    # Initialize storage
    clean_chunk = []
    noisy_chunk = []
    total_generated = 0
    
    # Open files for appending
    clean_fp = open(clean_file, 'wb')
    noisy_fp = open(noisy_file, 'wb')
    
    print(f"\n🚀 Starting data generation...")
    print(f"   Saving to: {clean_file.name} and {noisy_file.name}")
    
    with tqdm(total=target_samples, unit='samples', unit_scale=True) as pbar:
        text_idx = 0
        
        while total_generated < target_samples:
            # Get next text (cycle through)
            text = texts[text_idx % len(texts)]
            text_idx += 1
            
            # Generate clean OFDM waveform
            try:
                clean_waveform, meta = transceiver.transmit(text.encode('utf-8'))
            except Exception as e:
                print(f"\n⚠️  Skipping text due to error: {e}")
                continue
            
            # Generate multiple noisy versions at different SNRs
            for snr_db in snr_levels:
                noisy_waveform = add_awgn_noise(clean_waveform, snr_db)
                
                # Add to chunks
                clean_chunk.extend(clean_waveform)
                noisy_chunk.extend(noisy_waveform)
                
                total_generated += len(clean_waveform)
                pbar.update(len(clean_waveform))
                
                # Save chunk to disk if interval reached
                if len(clean_chunk) >= chunk_save_interval:
                    # Convert to numpy and save
                    clean_array = np.array(clean_chunk, dtype=np.complex64)
                    noisy_array = np.array(noisy_chunk, dtype=np.complex64)
                    
                    clean_array.tofile(clean_fp)
                    noisy_array.tofile(noisy_fp)
                    
                    # Clear chunks
                    clean_chunk = []
                    noisy_chunk = []
                    
                    # Flush to disk
                    clean_fp.flush()
                    noisy_fp.flush()
                
                # Stop if target reached
                if total_generated >= target_samples:
                    break
            
            if total_generated >= target_samples:
                break
    
    # Save remaining data
    if len(clean_chunk) > 0:
        print(f"\n💾 Saving final chunk ({len(clean_chunk):,} samples)...")
        clean_array = np.array(clean_chunk, dtype=np.complex64)
        noisy_array = np.array(noisy_chunk, dtype=np.complex64)
        
        clean_array.tofile(clean_fp)
        noisy_array.tofile(noisy_fp)
    
    # Close files
    clean_fp.close()
    noisy_fp.close()
    
    print("\n" + "="*80)
    print("✅ DATA GENERATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated Files:")
    print(f"   Clean: {clean_file} ({total_generated:,} samples)")
    print(f"   Noisy: {noisy_file} ({total_generated:,} samples)")
    print(f"   Total Size: {(total_generated * 8 * 2) / 1024**3:.2f} GB")
    
    # Verify files
    print(f"\n🔍 Verifying files...")
    clean_data = np.fromfile(clean_file, dtype=np.complex64)
    noisy_data = np.fromfile(noisy_file, dtype=np.complex64)
    
    print(f"   Clean samples: {len(clean_data):,}")
    print(f"   Noisy samples: {len(noisy_data):,}")
    print(f"   Clean power: {np.mean(np.abs(clean_data)**2):.2f}")
    print(f"   Match: {'✅' if len(clean_data) == len(noisy_data) else '❌'}")
    
    print("\n" + "="*80)
    print("🎓 READY FOR TRAINING!")
    print("="*80)
    print("\nNext step:")
    print(f"   python src/ofdm/model/train_ofdm.py \\")
    print(f"       --clean-data {clean_file} \\")
    print(f"       --noisy-data {noisy_file} \\")
    print(f"       --epochs 100")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Generate OFDM training data')
    
    parser.add_argument('--samples', type=int, default=100_000_000,
                       help='Number of IQ samples to generate (default: 100M)')
    parser.add_argument('--snr-min', type=float, default=0,
                       help='Minimum SNR in dB (default: 0)')
    parser.add_argument('--snr-max', type=float, default=25,
                       help='Maximum SNR in dB (default: 25)')
    parser.add_argument('--output-dir', type=str, default='dataset/OFDM',
                       help='Output directory (default: dataset/OFDM)')
    parser.add_argument('--chunk-size', type=int, default=10_000_000,
                       help='Save to disk every N samples (default: 10M)')
    
    args = parser.parse_args()
    
    generate_dataset(
        target_samples=args.samples,
        snr_range=(args.snr_min, args.snr_max),
        output_dir=args.output_dir,
        chunk_save_interval=args.chunk_size
    )


if __name__ == "__main__":
    main()
