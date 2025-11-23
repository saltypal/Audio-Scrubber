
"""
New Test Script using the Robust OFDM Library (src/ofdm/lib)
"""
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ofdm.lib.transceiver import OFDMTransmitter, OFDMReceiver
from src.ofdm.lib.utils import add_noise, plot_comparison
from src.ofdm.model.neuralnet import OFDM_UNet

def run_test():
    print("🚀 Starting Robust OFDM System Test...")
    
    # 1. Setup
    tx = OFDMTransmitter()
    rx = OFDMReceiver()
    
    text = "Hello! This is a test of the new robust OFDM library. It includes channel equalization to handle AI distortions."
    print(f"📝 Sending: '{text}'")
    
    # 2. Transmit
    clean_waveform, meta = tx.transmit(text.encode('utf-8'))
    print(f"📡 Generated Waveform: {len(clean_waveform)} samples, Power: {meta['raw_power']:.2f} -> Scaled to {np.mean(np.abs(clean_waveform)**2):.2f}")
    
    # 3. Channel (Add Noise)
    snr = 20  # High SNR to verify logic
    noisy_waveform = add_noise(clean_waveform, snr)
    print(f"📉 Added Noise (SNR {snr}dB)")
    
    # 4. AI Denoising
    print("🧠 Running AI Denoising...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OFDM_UNet().to(device)
    
    # Load Model
    model_path = "saved_models/OFDM/ofdm_final_1dunet.pth"
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model Loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Denoise in chunks
    denoised_waveform = np.zeros_like(noisy_waveform)
    chunk_size = 1024
    
    # Pad if needed
    pad_len = (chunk_size - (len(noisy_waveform) % chunk_size)) % chunk_size
    padded_noisy = np.pad(noisy_waveform, (0, pad_len))
    
    with torch.no_grad():
        for i in range(0, len(padded_noisy), chunk_size):
            chunk = padded_noisy[i:i+chunk_size]
            
            # To Tensor
            tensor = torch.stack([
                torch.from_numpy(chunk.real.astype(np.float32)),
                torch.from_numpy(chunk.imag.astype(np.float32))
            ]).unsqueeze(0).to(device)
            
            # Infer
            out = model(tensor)
            
            # To Numpy
            out_np = out.cpu().numpy()[0]
            denoised_chunk = out_np[0] + 1j*out_np[1]
            
            if i + chunk_size <= len(denoised_waveform):
                denoised_waveform[i:i+chunk_size] = denoised_chunk
            else:
                remain = len(denoised_waveform) - i
                denoised_waveform[i:] = denoised_chunk[:remain]

    # 5. Receive (Decode)
    print("\n📥 Decoding Results:")
    
    # Decode Noisy
    payload_noisy, info_noisy = rx.receive(noisy_waveform)
    print(f"   [Noisy Info]      : {info_noisy}")
    try:
        text_noisy = payload_noisy.decode('utf-8')
    except:
        text_noisy = "<Binary Garbage>"
    print(f"   [Noisy Channel]   : {text_noisy}")
    
    # Decode Denoised
    payload_denoised, info_denoised = rx.receive(denoised_waveform)
    print(f"   [Denoised Info]   : {info_denoised}")
    try:
        text_denoised = payload_denoised.decode('utf-8')
    except:
        text_denoised = "<Binary Garbage>"
    print(f"   [AI Denoised]     : {text_denoised}")
    
    # 6. Save Plot
    plot_comparison(clean_waveform, noisy_waveform, denoised_waveform, "output/new_lib_test.png")
    print("\n📊 Plot saved to output/new_lib_test.png")

if __name__ == "__main__":
    run_test()
