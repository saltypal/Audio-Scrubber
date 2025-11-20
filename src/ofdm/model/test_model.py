"""
Test the trained OFDM denoising model on sample data.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ofdm.model.neuralnet import OFDM_UNet

def test_model(model_path="saved_models/OFDM/unet1d_best.pth"):
    """Test the trained model on random QPSK data."""
    
    print("=" * 60)
    print("         OFDM MODEL TESTER")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Load model
    model = OFDM_UNet().to(device)
    
    if not Path(model_path).exists():
        print(f"\n❌ Model not found: {model_path}")
        print("\n💡 Train the model first:")
        print("   python src/ofdm/model/train_ofdm.py")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Loaded model: {model_path}")
    
    # Generate test QPSK signal
    num_samples = 1024
    print(f"\n🎲 Generating test QPSK signal ({num_samples} samples)...")
    
    # Random QPSK symbols
    qpsk_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j])
    clean_signal = qpsk_symbols[np.random.randint(0, 4, num_samples)]
    
    # Add noise (SNR = 10 dB)
    signal_power = np.mean(np.abs(clean_signal) ** 2)
    snr_db = 10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_power / 2)
    noisy_signal = clean_signal + noise
    
    print(f"   Clean signal power: {signal_power:.4f}")
    print(f"   Noise power: {noise_power:.4f}")
    print(f"   Input SNR: {snr_db} dB")
    
    # Prepare tensor
    noisy_tensor = torch.stack([
        torch.from_numpy(noisy_signal.real),
        torch.from_numpy(noisy_signal.imag)
    ]).float().unsqueeze(0).to(device)  # Add batch dimension
    
    # Denoise
    print("\n🧠 Running inference...")
    with torch.no_grad():
        output = model(noisy_tensor)
    
    # Convert back to complex
    output_np = output.cpu().numpy()[0]  # Remove batch dim
    denoised_signal = output_np[0] + 1j * output_np[1]
    
    # Calculate output SNR
    output_noise = denoised_signal - clean_signal
    output_noise_power = np.mean(np.abs(output_noise) ** 2)
    output_snr_db = 10 * np.log10(signal_power / output_noise_power)
    
    print(f"\n📊 Results:")
    print(f"   Output SNR: {output_snr_db:.2f} dB")
    print(f"   SNR Improvement: {output_snr_db - snr_db:.2f} dB")
    
    # Plot comparison
    plot_comparison(clean_signal, noisy_signal, denoised_signal)
    
    print("\n✅ Test complete!")
    print("=" * 60)

def plot_comparison(clean, noisy, denoised):
    """Plot clean, noisy, and denoised signals."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Time domain - I channel
    axes[0, 0].plot(clean.real[:200], label='Clean', linewidth=2, alpha=0.7)
    axes[0, 0].plot(noisy.real[:200], label='Noisy', linewidth=1, alpha=0.5)
    axes[0, 0].plot(denoised.real[:200], label='Denoised', linewidth=2, alpha=0.7)
    axes[0, 0].set_title('I Channel (Time Domain)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Time domain - Q channel
    axes[0, 1].plot(clean.imag[:200], label='Clean', linewidth=2, alpha=0.7)
    axes[0, 1].plot(noisy.imag[:200], label='Noisy', linewidth=1, alpha=0.5)
    axes[0, 1].plot(denoised.imag[:200], label='Denoised', linewidth=2, alpha=0.7)
    axes[0, 1].set_title('Q Channel (Time Domain)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Constellation - Clean
    axes[0, 2].scatter(clean.real, clean.imag, alpha=0.3, s=10, label='Clean')
    axes[0, 2].set_title('Constellation: Clean', fontweight='bold')
    axes[0, 2].set_xlabel('I')
    axes[0, 2].set_ylabel('Q')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axis('equal')
    
    # Constellation - Noisy
    axes[1, 0].scatter(noisy.real, noisy.imag, alpha=0.3, s=10, c='orange', label='Noisy')
    axes[1, 0].set_title('Constellation: Noisy', fontweight='bold')
    axes[1, 0].set_xlabel('I')
    axes[1, 0].set_ylabel('Q')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    
    # Constellation - Denoised
    axes[1, 1].scatter(denoised.real, denoised.imag, alpha=0.3, s=10, c='green', label='Denoised')
    axes[1, 1].set_title('Constellation: Denoised', fontweight='bold')
    axes[1, 1].set_xlabel('I')
    axes[1, 1].set_ylabel('Q')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    
    # Error comparison
    noisy_error = np.abs(noisy - clean)
    denoised_error = np.abs(denoised - clean)
    axes[1, 2].plot(noisy_error[:200], label='Noisy Error', linewidth=1, alpha=0.7)
    axes[1, 2].plot(denoised_error[:200], label='Denoised Error', linewidth=1, alpha=0.7)
    axes[1, 2].set_title('Absolute Error', fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path("saved_models/OFDM/test_results.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n📊 Visualization saved: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    test_model()
