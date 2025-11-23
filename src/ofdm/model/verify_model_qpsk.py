"""
Verify Model Works on Pure QPSK Symbols
This will show whether the model can denoise raw QPSK constellation points.
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ofdm.model.neuralnet import OFDM_UNet
from src.ofdm.model.qpsk_engine import QPSKModulator, ChannelSimulator


def generate_random_qpsk(num_symbols, power=35.0):
    """Generate random QPSK symbols at specified power."""
    # Random bits
    bits = np.random.randint(0, 2, num_symbols * 2)
    
    # Convert to QPSK
    symbols = QPSKModulator.bits_to_symbols(bits)
    
    # Scale to target power
    symbols = ChannelSimulator.normalize_power(symbols, power)
    
    return symbols, bits


def test_model_on_pure_qpsk():
    """
    Test if model can denoise pure QPSK constellation points.
    This is fundamentally different from GNU Radio OFDM signals.
    """
    
    print("=" * 70)
    print("    QPSK SYMBOL DENOISING TEST")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load model
    model_path = "saved_models/OFDM/ofdm_final_1dunet.pth"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    model = OFDM_UNet().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('best_val_loss', 0)
        print(f"✅ Model loaded: Epoch {epoch}, Val Loss: {val_loss:.6f}\n")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Model loaded\n")
    
    model.eval()
    
    # Test parameters
    chunk_size = 1024
    num_tests = 10
    snr_db = 10
    
    print(f"🧪 Test Configuration:")
    print(f"   Chunk size: {chunk_size} symbols")
    print(f"   Number of tests: {num_tests}")
    print(f"   SNR: {snr_db} dB")
    print(f"   Signal power: ~35 (matching training data)\n")
    
    print(f"🧠 Testing pure QPSK constellation points...\n")
    
    improvements = []
    ber_improvements = []
    
    for i in range(num_tests):
        # Generate random QPSK at training data power level
        clean_symbols, clean_bits = generate_random_qpsk(chunk_size, power=35.0)
        
        # Add noise
        noisy_symbols = ChannelSimulator.awgn(clean_symbols, snr_db)
        
        # Prepare input [1, 2, chunk_size]
        input_tensor = torch.stack([
            torch.from_numpy(noisy_symbols.real.astype(np.float32)),
            torch.from_numpy(noisy_symbols.imag.astype(np.float32))
        ]).unsqueeze(0).to(device)
        
        # Denoise
        with torch.no_grad():
            output = model(input_tensor)
        
        denoised = output[0, 0].cpu().numpy() + 1j * output[0, 1].cpu().numpy()
        
        # Calculate MSE
        mse_before = np.mean(np.abs(noisy_symbols - clean_symbols) ** 2)
        mse_after = np.mean(np.abs(denoised - clean_symbols) ** 2)
        mse_improvement = (mse_before - mse_after) / mse_before * 100
        
        # Calculate BER
        noisy_bits = QPSKModulator.symbols_to_bits(noisy_symbols)
        denoised_bits = QPSKModulator.symbols_to_bits(denoised)
        
        ber_before = np.sum(noisy_bits[:len(clean_bits)] != clean_bits) / len(clean_bits)
        ber_after = np.sum(denoised_bits[:len(clean_bits)] != clean_bits) / len(clean_bits)
        
        if ber_before > 0:
            ber_improvement = (ber_before - ber_after) / ber_before * 100
        else:
            ber_improvement = 0 if ber_after == 0 else -100
        
        improvements.append(mse_improvement)
        ber_improvements.append(ber_improvement)
        
        print(f"   Test {i+1:2d}:")
        print(f"      MSE:  {mse_before:.2f} → {mse_after:.2f} ({mse_improvement:+.1f}%)")
        print(f"      BER:  {ber_before:.6f} → {ber_after:.6f} ({ber_improvement:+.1f}%)")
    
    avg_mse_improvement = np.mean(improvements)
    avg_ber_improvement = np.mean(ber_improvements)
    
    print(f"\n📊 Results:")
    print(f"   Average MSE improvement: {avg_mse_improvement:.2f}%")
    print(f"   Average BER improvement: {avg_ber_improvement:.2f}%")
    print(f"   Best MSE: {max(improvements):.2f}%")
    print(f"   Worst MSE: {min(improvements):.2f}%")
    
    print(f"\n" + "=" * 70)
    
    if avg_mse_improvement > 0 and avg_ber_improvement > 0:
        print("✅ MODEL WORKS ON PURE QPSK!")
        print(f"   Average noise reduction: {avg_mse_improvement:.1f}%")
        print(f"   Average BER improvement: {avg_ber_improvement:.1f}%")
        print(f"\n💡 You can use this for raw QPSK symbol denoising.")
    elif avg_mse_improvement > 0 and avg_ber_improvement < 0:
        print("⚠️  MIXED RESULTS:")
        print(f"   MSE improved {avg_mse_improvement:.1f}% (signal cleaner)")
        print(f"   But BER got {abs(avg_ber_improvement):.1f}% worse (more bit errors)")
        print(f"\n💡 Model smooths noise but distorts constellation points.")
        print(f"   This is expected: trained on OFDM waveforms, not QPSK symbols.")
    else:
        print("❌ MODEL DOES NOT WORK ON PURE QPSK!")
        print(f"   MSE change: {avg_mse_improvement:+.1f}%")
        print(f"   BER change: {avg_ber_improvement:+.1f}%")
        print(f"\n💡 This is expected behavior:")
        print(f"   - Model was trained on GNU Radio OFDM signals")
        print(f"   - OFDM signals are complex waveforms (post-IFFT)")
        print(f"   - Pure QPSK symbols are discrete constellation points")
        print(f"   - These are fundamentally different signal types")
        print(f"\n   To denoise pure QPSK, you would need to:")
        print(f"   1. Generate new training data with raw QPSK symbols")
        print(f"   2. Retrain the model on that data")
    
    print("=" * 70)


if __name__ == "__main__":
    test_model_on_pure_qpsk()
