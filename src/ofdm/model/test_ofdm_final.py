"""
Test OFDM Text Transmission with AI Denoising
Uses complete OFDM engine that matches training data format
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ofdm.model.neuralnet import OFDM_UNet
from src.ofdm.model.ofdm_engine import (
    OFDMTransmitter, OFDMReceiver, ChannelSimulator, BERCalculator
)


def denoise_ofdm(model, noisy_waveform, device, chunk_size=1024):
    """Denoise OFDM waveform in chunks."""
    denoised = np.zeros_like(noisy_waveform)
    num_chunks = len(noisy_waveform) // chunk_size
    
    for i in range(num_chunks):
        chunk = noisy_waveform[i*chunk_size:(i+1)*chunk_size]
        
        # Prepare tensor [1, 2, chunk_size]
        chunk_tensor = torch.stack([
            torch.from_numpy(chunk.real.astype(np.float32)),
            torch.from_numpy(chunk.imag.astype(np.float32))
        ]).unsqueeze(0).to(device)
        
        # Denoise
        with torch.no_grad():
            output = model(chunk_tensor)
        
        # Convert back
        output_np = output.cpu().numpy()[0]
        denoised[i*chunk_size:(i+1)*chunk_size] = output_np[0] + 1j * output_np[1]
    
    # Handle remainder
    if len(noisy_waveform) % chunk_size != 0:
        remainder = noisy_waveform[num_chunks*chunk_size:]
        padded = np.pad(remainder, (0, chunk_size - len(remainder)), mode='constant')
        
        padded_tensor = torch.stack([
            torch.from_numpy(padded.real.astype(np.float32)),
            torch.from_numpy(padded.imag.astype(np.float32))
        ]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(padded_tensor)
        
        output_np = output.cpu().numpy()[0]
        denoised_remainder = output_np[0] + 1j * output_np[1]
        denoised[num_chunks*chunk_size:] = denoised_remainder[:len(remainder)]
    
    return denoised


def test_ofdm_text(model_path="saved_models/OFDM/ofdm_final_1dunet.pth",
                   test_text=None,
                   snr_db=10):
    """
    Complete OFDM text transmission test with AI denoising.
    This FINALLY uses the correct OFDM format!
    """
    
    if test_text is None:
        test_text = ("Hello OFDM World! This is a proper test using full OFDM modulation. "
                    "The quick brown fox jumps over the lazy dog. "
                    "Numbers: 0123456789. Special chars: @#$%^&*()")
    
    print("=" * 70)
    print("    OFDM TEXT TRANSMISSION TEST (Real OFDM Format)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load model
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    model = OFDM_UNet().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded (Epoch {checkpoint.get('epoch', '?')})\n")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Model loaded\n")
    
    model.eval()
    
    # === TRANSMIT ===
    print(f"📝 Original Text ({len(test_text)} chars):")
    print("-" * 70)
    print(test_text)
    print("-" * 70 + "\n")
    
    tx = OFDMTransmitter()
    clean_waveform, tx_meta = tx.transmit_text(test_text)
    
    print(f"📊 Transmission Stats:")
    print(f"   Payload: {tx_meta['payload_bytes']} bytes")
    print(f"   QPSK symbols: {tx_meta['qpsk_symbols']}")
    print(f"   OFDM samples: {tx_meta['ofdm_samples']}")
    print(f"   Signal power: {tx_meta['signal_power']:.2f}")
    print(f"   ✅ Scaled to match training data (~35)\n")
    
    # === CHANNEL ===
    noisy_waveform = ChannelSimulator.awgn(clean_waveform, snr_db)
    
    print(f"📡 Channel:")
    print(f"   SNR: {snr_db} dB")
    print(f"   Noisy power: {np.mean(np.abs(noisy_waveform)**2):.2f}\n")
    
    # === DENOISE ===
    print("🧠 AI Denoising OFDM waveform...")
    denoised_waveform = denoise_ofdm(model, noisy_waveform, device)
    
    # Calculate SNR improvement
    noise_before = noisy_waveform - clean_waveform
    noise_after = denoised_waveform - clean_waveform
    snr_improvement = 10 * np.log10(
        np.mean(np.abs(noise_before) ** 2) / np.mean(np.abs(noise_after) ** 2)
    )
    print(f"   SNR Improvement: {snr_improvement:.2f} dB\n")
    
    # === RECEIVE ===
    print("📥 Decoding OFDM...")
    rx = OFDMReceiver()
    
    text_noisy, noisy_meta = rx.receive_text(noisy_waveform)
    text_denoised, denoised_meta = rx.receive_text(denoised_waveform)
    
    # === METRICS ===
    ber_noisy = BERCalculator.calculate_ber(clean_waveform, noisy_waveform)
    ber_denoised = BERCalculator.calculate_ber(clean_waveform, denoised_waveform)
    
    print(f"\n📊 Results:")
    print(f"   BER (noisy):    {ber_noisy:.6f} ({ber_noisy*100:.4f}%)")
    print(f"   BER (denoised): {ber_denoised:.6f} ({ber_denoised*100:.4f}%)")
    
    if ber_noisy > 0:
        improvement = (ber_noisy - ber_denoised) / ber_noisy * 100
        print(f"   BER Improvement: {improvement:.2f}%")
    
    print(f"\n📝 Decoded Text (Noisy):")
    print("-" * 70)
    print(text_noisy)
    print("-" * 70)
    
    print(f"\n📝 Decoded Text (Denoised):")
    print("-" * 70)
    print(text_denoised)
    print("-" * 70)
    
    # Check if text matches
    if text_denoised.strip() == test_text.strip():
        print(f"\n🎉 SUCCESS! Text perfectly recovered after denoising!")
    elif text_noisy.strip() == test_text.strip():
        print(f"\n✅ Noisy channel was clean enough (no AI needed)")
    else:
        print(f"\n⚠️  Text has errors. Try higher SNR or check OFDM sync.")
    
    # === VISUALIZATION ===
    output_dir = Path("saved_models/OFDM/test_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Time domain (first 320 samples = 4 OFDM symbols)
    sample_count = min(320, len(clean_waveform))
    
    for idx, (signal, title) in enumerate([
        (clean_waveform[:sample_count], "Clean OFDM Waveform"),
        (noisy_waveform[:sample_count], "Noisy OFDM Waveform"),
        (denoised_waveform[:sample_count], "Denoised OFDM Waveform")
    ]):
        axes[0, idx].plot(signal.real, label='I', alpha=0.7, linewidth=0.8)
        axes[0, idx].plot(signal.imag, label='Q', alpha=0.7, linewidth=0.8)
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].set_title(title, fontweight='bold')
        axes[0, idx].set_xlabel('Sample')
        axes[0, idx].legend()
    
    # Spectrum
    for idx, (signal, title) in enumerate([
        (clean_waveform, "Clean Spectrum"),
        (noisy_waveform, "Noisy Spectrum"),
        (denoised_waveform, "Denoised Spectrum")
    ]):
        fft = np.fft.fftshift(np.fft.fft(signal[:1024]))
        axes[1, idx].plot(10 * np.log10(np.abs(fft) + 1e-10))
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].set_title(title, fontweight='bold')
        axes[1, idx].set_xlabel('Frequency Bin')
        axes[1, idx].set_ylabel('Power (dB)')
    
    plt.tight_layout()
    plot_path = output_dir / "ofdm_text_test.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n📊 Plot saved: {plot_path}")
    plt.close()
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run with default settings
    test_ofdm_text()
    
    # Or customize:
    # test_ofdm_text(test_text="Your message!", snr_db=15)
