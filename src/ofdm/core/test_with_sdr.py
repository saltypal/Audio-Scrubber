"""
Test OFDM AI Denoising with Real SDR Hardware
Works WITHOUT osmosdr - uses IQ file interchange
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ofdm.core.ofdm_pipeline import OFDMTransceiver, add_awgn_noise
from PIL import Image
import io

# Try to import AI model
try:
    from src.ofdm.model.neuralnet import OFDM_UNet
    import torch
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️  AI Model not available")


def denoise_waveform_with_ai(waveform: np.ndarray, model_path: str) -> np.ndarray:
    """Apply AI denoising to OFDM waveform"""
    if not MODEL_AVAILABLE:
        return waveform
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OFDM_UNet(in_channels=2, out_channels=2).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Normalize
    input_power = np.mean(np.abs(waveform)**2)
    normalized = waveform / np.sqrt(input_power + 1e-10)
    
    # Process in chunks
    chunk_size = 1024
    pad_len = (chunk_size - (len(normalized) % chunk_size)) % chunk_size
    padded = np.pad(normalized, (0, pad_len), mode='constant')
    
    denoised_chunks = []
    with torch.no_grad():
        for i in range(0, len(padded), chunk_size):
            chunk = padded[i:i+chunk_size]
            input_tensor = torch.stack([
                torch.from_numpy(np.real(chunk)),
                torch.from_numpy(np.imag(chunk))
            ]).unsqueeze(0).float().to(device)
            
            output = model(input_tensor)
            out_i = output[0, 0].cpu().numpy()
            out_q = output[0, 1].cpu().numpy()
            denoised_chunks.append(out_i + 1j*out_q)
    
    denoised = np.concatenate(denoised_chunks)[:len(normalized)]
    
    # Restore power
    denoised_power = np.mean(np.abs(denoised)**2)
    denoised = denoised * np.sqrt(input_power / (denoised_power + 1e-10))
    
    return denoised.astype(np.complex64)


def main():
    print("="*80)
    print(" "*20 + "OFDM SDR TEST - NO OSMOSDR NEEDED!")
    print("="*80)
    
    # Create test image
    print("\n📸 Creating test image...")
    test_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img_buffer = io.BytesIO()
    Image.fromarray(test_image).save(img_buffer, format='PNG')
    test_data = img_buffer.getvalue()
    
    print(f"   Data size: {len(test_data)} bytes")
    
    # Initialize transceiver
    transceiver = OFDMTransceiver()
    
    # Step 1: Generate TX waveform
    print("\n📡 Step 1: Generate OFDM waveform for transmission")
    clean_waveform, tx_meta = transceiver.transmit(test_data)
    
    print(f"   Generated {len(clean_waveform)} IQ samples")
    print(f"   OFDM symbols: {tx_meta['num_ofdm_symbols']}")
    print(f"   Power: {tx_meta['power_after_scale']:.2f}")
    
    # Save for Pluto TX
    output_dir = Path('dataset/OFDM')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tx_file = output_dir / 'tx_waveform.iq'
    clean_waveform.tofile(tx_file)
    print(f"\n✅ Saved TX waveform: {tx_file}")
    print(f"   Use this with Pluto TX in GNU Radio!")
    
    # Simulate: Add noise (in real scenario, this happens over the air)
    print("\n📻 Step 2: Simulate channel (add 10dB noise)")
    noisy_waveform = add_awgn_noise(clean_waveform, snr_db=10)
    
    # Save simulated RX
    rx_file = output_dir / 'rx_waveform_noisy.iq'
    noisy_waveform.tofile(rx_file)
    print(f"✅ Saved noisy RX: {rx_file}")
    
    # Test without AI
    print("\n🔍 Step 3: Test RX without AI denoising")
    data_noisy, meta_noisy = transceiver.receive(noisy_waveform)
    
    if data_noisy:
        try:
            img_noisy = np.array(Image.open(io.BytesIO(data_noisy)))
            print(f"✅ Decoded successfully!")
            print(f"   Packets: {meta_noisy['total_packets']}")
            print(f"   CRC errors: {meta_noisy['crc_errors']}")
            print(f"   Error rate: {meta_noisy['packet_error_rate']*100:.1f}%")
        except:
            print(f"❌ Image decode failed")
    else:
        print(f"❌ RX failed: {meta_noisy}")
    
    # Test with AI
    if MODEL_AVAILABLE:
        print("\n🧠 Step 4: Test RX WITH AI denoising")
        
        model_path = 'saved_models/OFDM/ofdm_final_1dunet.pth'
        if not Path(model_path).exists():
            model_path = 'saved_models/OFDM/unet1d_best.pth'
        
        if Path(model_path).exists():
            print(f"   Loading model: {Path(model_path).name}")
            denoised_waveform = denoise_waveform_with_ai(noisy_waveform, model_path)
            
            # Save denoised waveform
            denoised_file = output_dir / 'rx_waveform_denoised.iq'
            denoised_waveform.tofile(denoised_file)
            print(f"✅ Saved denoised: {denoised_file}")
            
            # Decode
            data_clean, meta_clean = transceiver.receive(denoised_waveform)
            
            if data_clean:
                try:
                    img_clean = np.array(Image.open(io.BytesIO(data_clean)))
                    print(f"✅ AI denoised - Decoded successfully!")
                    print(f"   Packets: {meta_clean['total_packets']}")
                    print(f"   CRC errors: {meta_clean['crc_errors']}")
                    print(f"   Error rate: {meta_clean['packet_error_rate']*100:.1f}%")
                except:
                    print(f"❌ Image decode failed after AI")
            else:
                print(f"❌ AI RX failed: {meta_clean}")
        else:
            print(f"⚠️  Model not found at {model_path}")
    
    print("\n" + "="*80)
    print("WORKFLOW FOR REAL SDR:")
    print("="*80)
    print("1. TX: Load 'tx_waveform.iq' into GNU Radio → Pluto TX")
    print("2. RX: Capture with RTL-SDR → Save as 'rx_captured.iq'")
    print("3. AI: Run AI denoising on rx_captured.iq")
    print("4. Decode: Use this script to decode the denoised file")
    print("="*80)


if __name__ == "__main__":
    main()
