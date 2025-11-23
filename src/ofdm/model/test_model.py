"""
Test the trained OFDM denoising model on real PNG file transmission.
Follows the workflow: File → QPSK → Add Noise → Denoise → Reconstruct
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ofdm.model.neuralnet import OFDM_UNet
from src.ofdm.TxRx.sdr_hardware import SignalUtils

def test_model(model_path="saved_models/OFDM/final_models/ofdm_1dunet.pth", 
               image_path="src/ofdm/TxRx/Transcorn/testfile_img.png"):
    """Test the trained model on real PNG file following Tx/Rx workflow."""
    
    print("=" * 60)
    print("    OFDM MODEL TESTER (Real File Workflow)")
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
    
    # Load checkpoint (contains model_state_dict + metadata)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract just the model weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from checkpoint: {model_path}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
    else:
        # Old format - direct state dict
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded model: {model_path}")
    
    model.eval()
    
    # === STEP 1: File → QPSK ===
    print(f"\n📁 Loading file: {image_path}")
    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        return
    
    clean_signal, metadata = SignalUtils.file_to_qpsk(image_path)
    print(f"   ✅ Converted to QPSK: {len(clean_signal)} symbols")
    print(f"   📝 Metadata: {metadata['filename']} ({metadata['size']} bytes)")
    
    # === STEP 2: Add Noise (Simulate Rx) ===
    signal_power = np.mean(np.abs(clean_signal) ** 2)
    snr_db = 10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = (np.random.randn(len(clean_signal)) + 1j * np.random.randn(len(clean_signal))) * np.sqrt(noise_power / 2)
    noisy_signal = clean_signal + noise
    
    print(f"\n📡 Simulating noisy reception:")
    print(f"   Clean signal power: {signal_power:.4f}")
    print(f"   Noise power: {noise_power:.4f}")
    print(f"   Input SNR: {snr_db} dB")
    
    # === STEP 3: Denoise with AI ===
    print("\n🧠 Running AI denoising...")
    denoised_signal = denoise_signal_chunked(model, noisy_signal, device)
    
    # Calculate output SNR
    output_noise = denoised_signal - clean_signal
    output_noise_power = np.mean(np.abs(output_noise) ** 2)
    output_snr_db = 10 * np.log10(signal_power / output_noise_power)
    
    print(f"\n📊 Denoising Results:")
    print(f"   Output SNR: {output_snr_db:.2f} dB")
    print(f"   SNR Improvement: {output_snr_db - snr_db:.2f} dB")
    
    # === STEP 4: Reconstruct Files ===
    print("\n🔧 Reconstructing images...")
    output_dir = Path("saved_models/OFDM/test_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save clean (original)
    clean_path = output_dir / "1_clean_original.png"
    SignalUtils.qpsk_to_file(clean_signal, clean_path)
    
    # Save noisy
    noisy_path = output_dir / "2_noisy_received.png"
    SignalUtils.qpsk_to_file(noisy_signal, noisy_path)
    
    # Save denoised
    denoised_path = output_dir / "3_denoised_ai.png"
    SignalUtils.qpsk_to_file(denoised_signal, denoised_path)
    
    print(f"   ✅ Saved: {clean_path}")
    print(f"   ✅ Saved: {noisy_path}")
    print(f"   ✅ Saved: {denoised_path}")
    
    # === STEP 5: Plot Comparison ===
    plot_comparison(clean_path, noisy_path, denoised_path, 
                   clean_signal, noisy_signal, denoised_signal)
    
    print("\n✅ Test complete!")
    print("=" * 60)

def denoise_signal_chunked(model, signal, device, chunk_size=1024):
    """Denoise signal in chunks (same as SignalUtils.denoise_signal but local)."""
    denoised = np.zeros_like(signal)
    num_chunks = len(signal) // chunk_size
    
    for i in range(num_chunks):
        chunk = signal[i*chunk_size:(i+1)*chunk_size]
        
        # Prepare tensor
        chunk_tensor = torch.stack([
            torch.from_numpy(chunk.real),
            torch.from_numpy(chunk.imag)
        ]).float().unsqueeze(0).to(device)
        
        # Denoise
        with torch.no_grad():
            output = model(chunk_tensor)
        
        # Convert back
        output_np = output.cpu().numpy()[0]
        denoised[i*chunk_size:(i+1)*chunk_size] = output_np[0] + 1j * output_np[1]
    
    # Handle remainder
    if len(signal) % chunk_size != 0:
        remainder = signal[num_chunks*chunk_size:]
        padded = np.pad(remainder, (0, chunk_size - len(remainder)), mode='constant')
        
        padded_tensor = torch.stack([
            torch.from_numpy(padded.real),
            torch.from_numpy(padded.imag)
        ]).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(padded_tensor)
        
        output_np = output.cpu().numpy()[0]
        denoised_remainder = output_np[0] + 1j * output_np[1]
        denoised[num_chunks*chunk_size:] = denoised_remainder[:len(remainder)]
    
    return denoised

def plot_comparison(clean_path, noisy_path, denoised_path, 
                   clean_signal, noisy_signal, denoised_signal):
    """Plot image comparison and signal analysis."""
    fig = plt.figure(figsize=(18, 10))
    
    # Load images
    try:
        img_clean = Image.open(clean_path)
        img_noisy = Image.open(noisy_path)
        img_denoised = Image.open(denoised_path)
    except Exception as e:
        print(f"⚠️  Could not load images for plotting: {e}")
        return
    
    # === ROW 1: Images ===
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(img_clean)
    ax1.set_title('1. Original (Clean)', fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(img_noisy)
    ax2.set_title('2. After Noise (Rx)', fontweight='bold', fontsize=12)
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(img_denoised)
    ax3.set_title('3. After AI Denoise', fontweight='bold', fontsize=12)
    ax3.axis('off')
    
    # === ROW 2: Constellations ===
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(clean_signal.real[:5000], clean_signal.imag[:5000], 
               alpha=0.2, s=5, c='blue')
    ax4.set_title('Clean Constellation', fontweight='bold')
    ax4.set_xlabel('I')
    ax4.set_ylabel('Q')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    ax4.set_xlim(-3, 3)
    ax4.set_ylim(-3, 3)
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(noisy_signal.real[:5000], noisy_signal.imag[:5000], 
               alpha=0.2, s=5, c='orange')
    ax5.set_title('Noisy Constellation', fontweight='bold')
    ax5.set_xlabel('I')
    ax5.set_ylabel('Q')
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    ax5.set_xlim(-3, 3)
    ax5.set_ylim(-3, 3)
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(denoised_signal.real[:5000], denoised_signal.imag[:5000], 
               alpha=0.2, s=5, c='green')
    ax6.set_title('Denoised Constellation', fontweight='bold')
    ax6.set_xlabel('I')
    ax6.set_ylabel('Q')
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')
    ax6.set_xlim(-3, 3)
    ax6.set_ylim(-3, 3)
    
    # === ROW 3: Time Domain & Error ===
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(clean_signal.real[:500], label='Clean', linewidth=2, alpha=0.7)
    ax7.plot(noisy_signal.real[:500], label='Noisy', linewidth=1, alpha=0.5)
    ax7.plot(denoised_signal.real[:500], label='Denoised', linewidth=2, alpha=0.7)
    ax7.set_title('I Channel (Time Domain)', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(clean_signal.imag[:500], label='Clean', linewidth=2, alpha=0.7)
    ax8.plot(noisy_signal.imag[:500], label='Noisy', linewidth=1, alpha=0.5)
    ax8.plot(denoised_signal.imag[:500], label='Denoised', linewidth=2, alpha=0.7)
    ax8.set_title('Q Channel (Time Domain)', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    ax9 = plt.subplot(3, 3, 9)
    noisy_error = np.abs(noisy_signal - clean_signal)
    denoised_error = np.abs(denoised_signal - clean_signal)
    ax9.plot(noisy_error[:500], label='Noisy Error', linewidth=1, alpha=0.7, c='orange')
    ax9.plot(denoised_error[:500], label='Denoised Error', linewidth=1, alpha=0.7, c='green')
    ax9.set_title('Absolute Error', fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path("saved_models/OFDM/test_results.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n📊 Visualization saved: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    test_model()
