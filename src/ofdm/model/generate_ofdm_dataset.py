"""
OFDM Training Dataset Generator - Optimized for Custom Pipeline
Generates clean/noisy OFDM waveform pairs with proper buffer alignment
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ofdm.core.ofdm_pipeline import OFDMTransceiver, add_awgn_noise

# GNU Radio minimum output buffer
MIN_BUFFER_SIZE = 65536


def generate_diverse_data(num_samples=5000):
    """Generate diverse binary data for training"""
    data_samples = []
    
    # 1. Text messages (varied lengths)
    texts = [
        b"OFDM test transmission with AI denoising capability",
        b"Wireless communication system - Status: ACTIVE",
        b"Hello World! Testing 123... Signal quality check.",
        b"Digital modulation QPSK with pilot-based equalization",
        b"ERROR: Packet loss detected. Retransmitting...",
        b"SUCCESS: All CRC checks passed. Ready for deployment.",
    ]
    
    for _ in range(1000):
        # Random text
        text = texts[np.random.randint(0, len(texts))]
        # Random repetitions
        data_samples.append(text * np.random.randint(1, 10))
    
    # 2. Random binary data (simulating images, files)
    for _ in range(2000):
        length = np.random.randint(100, 5000)
        data_samples.append(np.random.bytes(length))
    
    # 3. Structured data (simulating packets)
    for i in range(2000):
        packet = f"PKT{i:06d}|DATA:".encode() + np.random.bytes(np.random.randint(50, 500))
        data_samples.append(packet)
    
    return data_samples[:num_samples]


def generate_dataset(
    num_waveforms=1000,
    snr_min=0,
    snr_max=25,
    output_dir='dataset/OFDM',
    align_to_buffer=True
):
    """
    Generate OFDM training dataset with proper alignment
    
    Args:
        num_waveforms: Number of waveform pairs to generate
        snr_min: Minimum SNR in dB
        snr_max: Maximum SNR in dB
        output_dir: Output directory
        align_to_buffer: Align to GNU Radio buffer size (65536)
    """
    print("="*80)
    print("OFDM TRAINING DATASET GENERATOR - CUSTOM PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Waveforms: {num_waveforms}")
    print(f"  SNR Range: {snr_min} to {snr_max} dB")
    print(f"  Buffer Alignment: {align_to_buffer} ({MIN_BUFFER_SIZE if align_to_buffer else 'None'})")
    print(f"  Output: {output_dir}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output files
    clean_file = output_dir / 'train_clean.iq'
    noisy_file = output_dir / 'train_noisy.iq'
    
    # Initialize
    transceiver = OFDMTransceiver()
    data_samples = generate_diverse_data(num_waveforms)
    
    print(f"\n📝 Generated {len(data_samples)} diverse data samples")
    
    # Open files
    clean_fp = open(clean_file, 'wb')
    noisy_fp = open(noisy_file, 'wb')
    
    total_samples = 0
    clean_buffer = []
    noisy_buffer = []
    
    print(f"\n🚀 Generating training data...")
    
    with tqdm(total=num_waveforms, desc="Waveforms") as pbar:
        for i in range(num_waveforms):
            # Get data
            data = data_samples[i % len(data_samples)]
            
            # Generate clean waveform
            try:
                clean_waveform, _ = transceiver.transmit(data)
            except Exception as e:
                print(f"\n⚠️  Skipping sample {i}: {e}")
                continue
            
            # Random SNR for this waveform
            snr_db = np.random.uniform(snr_min, snr_max)
            noisy_waveform = add_awgn_noise(clean_waveform, snr_db)
            
            # Add to buffers
            clean_buffer.extend(clean_waveform)
            noisy_buffer.extend(noisy_waveform)
            
            # Save when buffer is large enough
            if align_to_buffer and len(clean_buffer) >= MIN_BUFFER_SIZE:
                # Align to buffer size
                num_complete_buffers = len(clean_buffer) // MIN_BUFFER_SIZE
                samples_to_save = num_complete_buffers * MIN_BUFFER_SIZE
                
                # Save aligned portion
                clean_array = np.array(clean_buffer[:samples_to_save], dtype=np.complex64)
                noisy_array = np.array(noisy_buffer[:samples_to_save], dtype=np.complex64)
                
                clean_array.tofile(clean_fp)
                noisy_array.tofile(noisy_fp)
                
                total_samples += samples_to_save
                
                # Keep remainder
                clean_buffer = clean_buffer[samples_to_save:]
                noisy_buffer = noisy_buffer[samples_to_save:]
                
                # Flush
                clean_fp.flush()
                noisy_fp.flush()
            
            pbar.update(1)
            pbar.set_postfix({'Total_Samples': f'{total_samples:,}'})
    
    # Save remaining (may not be aligned)
    if len(clean_buffer) > 0:
        print(f"\n💾 Saving final {len(clean_buffer):,} samples...")
        clean_array = np.array(clean_buffer, dtype=np.complex64)
        noisy_array = np.array(noisy_buffer, dtype=np.complex64)
        clean_array.tofile(clean_fp)
        noisy_array.tofile(noisy_fp)
        total_samples += len(clean_buffer)
    
    clean_fp.close()
    noisy_fp.close()
    
    # Verify
    print("\n" + "="*80)
    print("✅ DATASET GENERATION COMPLETE!")
    print("="*80)
    
    clean_data = np.fromfile(clean_file, dtype=np.complex64)
    noisy_data = np.fromfile(noisy_file, dtype=np.complex64)
    
    print(f"\nDataset Statistics:")
    print(f"  Clean file: {clean_file.name}")
    print(f"  Noisy file: {noisy_file.name}")
    print(f"  Total samples: {len(clean_data):,}")
    print(f"  File size: {len(clean_data) * 8 / 1024**2:.2f} MB each")
    print(f"  Clean power: {np.mean(np.abs(clean_data)**2):.2f}")
    print(f"  Noisy power: {np.mean(np.abs(noisy_data)**2):.2f}")
    print(f"  Lengths match: {'✅' if len(clean_data) == len(noisy_data) else '❌'}")
    
    if align_to_buffer:
        alignment = len(clean_data) % MIN_BUFFER_SIZE
        print(f"  Buffer alignment: {alignment} samples offset (target: 0)")
    
    print(f"\n🎓 Ready for training!")
    print(f"   python src/ofdm/model/train_ofdm_optimized.py \\")
    print(f"       --clean-data {clean_file} \\")
    print(f"       --noisy-data {noisy_file}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Generate OFDM training dataset')
    parser.add_argument('--waveforms', type=int, default=1000,
                       help='Number of waveforms to generate (default: 1000)')
    parser.add_argument('--snr-min', type=float, default=0,
                       help='Minimum SNR in dB (default: 0)')
    parser.add_argument('--snr-max', type=float, default=25,
                       help='Maximum SNR in dB (default: 25)')
    parser.add_argument('--output-dir', type=str, default='dataset/OFDM',
                       help='Output directory (default: dataset/OFDM)')
    parser.add_argument('--no-align', action='store_true',
                       help='Disable buffer alignment')
    
    args = parser.parse_args()
    
    generate_dataset(
        num_waveforms=args.waveforms,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        output_dir=args.output_dir,
        align_to_buffer=not args.no_align
    )


if __name__ == "__main__":
    main()
