"""
CORRECT Test: Verify model works on actual training data format
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ofdm.model.neuralnet import OFDM_UNet


def test_model_on_training_data():
    """
    The ONLY valid test: Use actual training data chunks.
    Your model denoises GNU Radio OFDM signals, not raw QPSK.
    """
    
    print("=" * 70)
    print("    CORRECT MODEL TEST (Using Actual Training Data)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load model
    model_path = "saved_models/OFDM/ofdm_final_1dunet.pth"
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
    
    # Load training data
    print("📁 Loading training data...")
    clean_data = np.fromfile("dataset/OFDM/clean_ofdm.iq", dtype=np.complex64)
    noisy_data = np.fromfile("dataset/OFDM/noisy_ofdm.iq", dtype=np.complex64)
    
    print(f"   Samples: {len(clean_data):,}")
    print(f"   Clean power: {np.mean(np.abs(clean_data)**2):.2f}")
    print(f"   Noisy power: {np.mean(np.abs(noisy_data)**2):.2f}\n")
    
    # Test on random chunks
    chunk_size = 1024
    num_tests = 10
    
    print(f"🧠 Testing {num_tests} random chunks of {chunk_size} samples...\n")
    
    improvements = []
    
    for i in range(num_tests):
        idx = np.random.randint(0, len(clean_data) - chunk_size)
        
        clean_chunk = clean_data[idx:idx+chunk_size]
        noisy_chunk = noisy_data[idx:idx+chunk_size]
        
        # Prepare input
        input_tensor = torch.stack([
            torch.from_numpy(noisy_chunk.real.astype(np.float32)),
            torch.from_numpy(noisy_chunk.imag.astype(np.float32))
        ]).unsqueeze(0).to(device)
        
        # Denoise
        with torch.no_grad():
            output = model(input_tensor)
        
        denoised = output[0, 0].cpu().numpy() + 1j * output[0, 1].cpu().numpy()
        
        # Calculate MSE
        mse_before = np.mean(np.abs(noisy_chunk - clean_chunk) ** 2)
        mse_after = np.mean(np.abs(denoised - clean_chunk) ** 2)
        improvement = (mse_before - mse_after) / mse_before * 100
        
        improvements.append(improvement)
        
        print(f"   Test {i+1:2d}: MSE {mse_before:.2f} → {mse_after:.2f} "
              f"({improvement:+.1f}%)")
    
    avg_improvement = np.mean(improvements)
    
    print(f"\n📊 Results:")
    print(f"   Average MSE improvement: {avg_improvement:.2f}%")
    print(f"   Best: {max(improvements):.2f}%")
    print(f"   Worst: {min(improvements):.2f}%")
    
    if avg_improvement > 0:
        print(f"\n✅ MODEL WORKS! Average {avg_improvement:.1f}% noise reduction.")
        print(f"\n💡 For text transmission:")
        print(f"   1. Use GNU Radio OFDM Tx to generate signal from text")
        print(f"   2. Denoise the RAW IQ output (like USE_OFDM.py does)")
        print(f"   3. Use GNU Radio OFDM Rx to decode back to text")
        print(f"\n   Your model denoises OFDM waveforms, NOT raw QPSK symbols!")
    else:
        print(f"\n❌ Model isn't working properly. May need retraining.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_model_on_training_data()
