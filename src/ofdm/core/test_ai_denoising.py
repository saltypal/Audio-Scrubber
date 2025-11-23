"""
================================================================================
OFDM TEST WORKFLOW - CORRECT PIPELINE
================================================================================
This script demonstrates the CORRECT workflow:

1. Generate clean OFDM waveform from text
2. Add channel noise (AWGN)
3. Path A (Control): Demodulate noisy waveform → Get garbage
4. Path B (AI): Denoise waveform → Demodulate → Get clean text
5. Compare BER and Constellations

Key: AI processes TIME-DOMAIN OFDM WAVEFORM, not QPSK symbols!
================================================================================
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import io

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ofdm.core.ofdm_pipeline import (
    OFDMTransceiver, OFDMParams, calculate_ber, add_awgn_noise
)

try:
    from src.ofdm.model.neuralnet import OFDM_UNet
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️  AI Model not available - will skip denoising comparison")


def denoise_waveform(waveform: np.ndarray, model_path: str, chunk_size: int = 1024, params: OFDMParams = None) -> np.ndarray:
    """
    Apply AI denoising to OFDM waveform (time-domain IQ samples)
    
    Args:
        waveform: Noisy OFDM waveform (complex64)
        model_path: Path to trained 1D U-Net model
        chunk_size: Process in chunks to avoid memory issues
        params: OFDM parameters for pilot-based phase correction
        
    Returns:
        Denoised waveform (complex64)
    """
    if not MODEL_AVAILABLE:
        print("⚠️  Model not available, returning original waveform")
        return waveform
    
    if params is None:
        params = OFDMParams()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = OFDM_UNet(in_channels=2, out_channels=2).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Model loaded: {Path(model_path).name}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return waveform
    
    # Normalize input power (model expects normalized signals)
    input_power = np.mean(np.abs(waveform)**2)
    normalized_waveform = waveform / np.sqrt(input_power + 1e-10)
    
    # Pad to chunk size
    pad_len = (chunk_size - (len(normalized_waveform) % chunk_size)) % chunk_size
    padded = np.pad(normalized_waveform, (0, pad_len), mode='constant')
    
    denoised_chunks = []
    
    with torch.no_grad():
        for i in range(0, len(padded), chunk_size):
            chunk = padded[i:i+chunk_size]
            
            # Prepare input: [Batch=1, Channels=2, Length]
            input_tensor = torch.stack([
                torch.from_numpy(np.real(chunk)),
                torch.from_numpy(np.imag(chunk))
            ]).unsqueeze(0).float().to(device)
            
            # Inference
            output_tensor = model(input_tensor)
            
            # Extract I/Q
            out_i = output_tensor[0, 0].cpu().numpy()
            out_q = output_tensor[0, 1].cpu().numpy()
            denoised_chunks.append(out_i + 1j*out_q)
    
    denoised = np.concatenate(denoised_chunks)[:len(normalized_waveform)]
    
    # FIX 1 & 2: PILOT-BASED NORMALIZATION & PHASE CORRECTION
    # The key insight: normalize based on PILOT power, not overall signal power
    # This ensures channel estimator sees pilots at correct magnitude
    sym_len = params.symbol_length
    if len(denoised) >= sym_len:
        # Extract first OFDM symbol
        first_symbol = denoised[:sym_len]
        first_symbol_no_cp = first_symbol[params.cp_len:]
        freq_domain = np.fft.fft(first_symbol_no_cp)
        
        # Extract pilot carriers
        pilots_rx = []
        for carrier_idx in params.pilot_carriers:
            pilots_rx.append(freq_domain[carrier_idx % params.fft_size])
        pilots_rx = np.array(pilots_rx)
        
        # Calculate scaling based on pilot power
        pilot_power_rx = np.mean(np.abs(pilots_rx)**2)
        pilot_power_expected = np.mean(np.abs(params.pilot_values)**2)
        
        pilot_scale = np.sqrt(pilot_power_expected / (pilot_power_rx + 1e-10))
        
        # Apply pilot-based scaling
        denoised = denoised * pilot_scale
        
        print(f"   Pilot-based normalization:")
        print(f"      Pilot power (before): {pilot_power_rx:.4f}")
        print(f"      Expected pilot power: {pilot_power_expected:.4f}")
        print(f"      Scale factor applied: {pilot_scale:.4f}")
        print(f"      Pilot power (after): {np.mean(np.abs(pilots_rx * pilot_scale)**2):.4f}")
        
        # Now check phase alignment
        phase_errors = np.angle(pilots_rx * pilot_scale) - np.angle(params.pilot_values)
        avg_phase_error = np.mean(phase_errors)
        
        if abs(avg_phase_error) > 0.1:
            phase_correction = np.exp(-1j * avg_phase_error)
            denoised = denoised * phase_correction
            print(f"      Phase correction: {np.degrees(avg_phase_error):.2f}° removed")
    
    final_power = np.mean(np.abs(denoised)**2)
    print(f"   Final diagnostics:")
    print(f"      Signal power: {final_power:.4f}")
    print(f"      Avg magnitude: {np.mean(np.abs(denoised)):.4f}")
    
    return denoised.astype(np.complex64)


def run_test():
    """Main test workflow"""
    
    print("="*80)
    print(" "*20 + "OFDM DENOISING TEST - CORRECT PIPELINE")
    print("="*80)
    
    # Configuration
    params = OFDMParams()
    transceiver = OFDMTransceiver(params)
    
    # Create a small test image (32x32 RGB)
    print("\n🖼️  Creating test image (32x32 RGB)...")
    test_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    
    # Convert image to bytes
    img_buffer = io.BytesIO()
    Image.fromarray(test_image).save(img_buffer, format='PNG')
    message_bytes = img_buffer.getvalue()
    
    print(f"\n📝 Original Data ({len(message_bytes)} bytes):")
    print(f"   Image: 32x32 RGB PNG ({len(message_bytes)} bytes)")
    
    # Step 1: Generate Clean OFDM Waveform
    print("\n" + "─"*80)
    print("STEP 1: TRANSMITTER - Generate Clean OFDM Waveform")
    print("─"*80)
    
    clean_waveform, tx_meta = transceiver.transmit(message_bytes)
    
    print(f"✅ Generated clean OFDM waveform:")
    print(f"   Waveform Length: {len(clean_waveform)} samples")
    print(f"   OFDM Symbols: {tx_meta['num_ofdm_symbols']}")
    print(f"   QPSK Symbols: {tx_meta['num_qpsk_symbols']}")
    print(f"   Power: {tx_meta['power_after_scale']:.2f}")
    
    # Save original bits for BER calculation
    packet = transceiver.packet.encode(message_bytes)
    original_bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    
    # Step 2: Add Channel Noise
    print("\n" + "─"*80)
    print("STEP 2: CHANNEL - Add AWGN Noise (SNR = 10 dB)")
    print("─"*80)
    
    snr_db = 10
    noisy_waveform = add_awgn_noise(clean_waveform, snr_db)
    
    print(f"✅ Added noise:")
    print(f"   Target SNR: {snr_db} dB")
    print(f"   Noisy Waveform Length: {len(noisy_waveform)} samples")
    print(f"   Noisy power: {np.mean(np.abs(noisy_waveform)**2):.4f}")
    print(f"   Noisy avg magnitude: {np.mean(np.abs(noisy_waveform)):.4f}")
    
    # Step 3: Path A - Control (No AI)
    print("\n" + "─"*80)
    print("STEP 3A: RECEIVER (CONTROL) - Decode Without AI")
    print("─"*80)
    
    noisy_data, noisy_meta = transceiver.receive(noisy_waveform)
    
    noisy_image = None
    if noisy_data:
        try:
            # Try to reconstruct image
            noisy_img_buffer = io.BytesIO(noisy_data)
            noisy_image = np.array(Image.open(noisy_img_buffer))
            print(f"✅ Decoded (Noisy Channel):")
            print(f"   Image shape: {noisy_image.shape}")
            
            # Calculate BER
            noisy_packet = transceiver.packet.encode(noisy_data)
            noisy_bits = np.unpackbits(np.frombuffer(noisy_packet, dtype=np.uint8))
            ber_noisy = calculate_ber(original_bits, noisy_bits)
            print(f"   BER (Control): {ber_noisy:.4f} ({ber_noisy*100:.2f}%)")
        except Exception as e:
            print(f"❌ Image decode failed: {e}")
            ber_noisy = 1.0
    else:
        print(f"❌ Control Path Failed: {noisy_meta}")
        ber_noisy = 1.0
    
    # Step 4: Path B - AI Denoising
    print("\n" + "─"*80)
    print("STEP 3B: RECEIVER (AI) - Denoise Waveform → Decode")
    print("─"*80)
    
    if not MODEL_AVAILABLE:
        print("⚠️  Skipping AI path - model not available")
        denoised_waveform = noisy_waveform
        ber_denoised = ber_noisy
    else:
        # AI Denoising on WAVEFORM
        model_path = 'saved_models/OFDM/ofdm_final_1dunet.pth'
        if not Path(model_path).exists():
            model_path = 'saved_models/OFDM/unet1d_best.pth'
        
        print(f"🧠 Applying AI Denoising to OFDM waveform...")
        denoised_waveform = denoise_waveform(noisy_waveform, model_path, params=params)
        
        print(f"✅ Denoising complete")
        
        # FIX 2: DIAGNOSTIC - Check for phase rotation
        print(f"\n🔍 DIAGNOSTIC - Checking AI output constellation...")
        qpsk_denoised_diag, _ = transceiver.ofdm_demod.demodulate(denoised_waveform)
        avg_phase = np.angle(np.mean(qpsk_denoised_diag))
        phase_std = np.std(np.angle(qpsk_denoised_diag))
        print(f"   Average phase: {np.degrees(avg_phase):.2f}°")
        print(f"   Phase std dev: {np.degrees(phase_std):.2f}°")
        if abs(avg_phase) > 0.3:  # More than ~17 degrees rotation
            print(f"   ⚠️  WARNING: Significant phase rotation detected!")
        
        # Demodulate denoised waveform
        denoised_data, denoised_meta = transceiver.receive(denoised_waveform)
        
        denoised_image = None
        if denoised_data:
            try:
                # Try to reconstruct image
                denoised_img_buffer = io.BytesIO(denoised_data)
                denoised_image = np.array(Image.open(denoised_img_buffer))
                print(f"✅ Decoded (After AI):")
                print(f"   Image shape: {denoised_image.shape}")
                
                # Calculate BER
                denoised_packet = transceiver.packet.encode(denoised_data)
                denoised_bits = np.unpackbits(np.frombuffer(denoised_packet, dtype=np.uint8))
                ber_denoised = calculate_ber(original_bits, denoised_bits)
                print(f"   BER (AI): {ber_denoised:.4f} ({ber_denoised*100:.2f}%)")
            except Exception as e:
                print(f"❌ Image decode failed: {e}")
                ber_denoised = 1.0
        else:
            print(f"❌ AI Path Failed: {denoised_meta}")
            ber_denoised = 1.0
    
    # Step 5: Visualization
    print("\n" + "─"*80)
    print("STEP 4: VISUALIZATION - Waveforms, Constellations & Images")
    print("─"*80)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    
    # Row 1: Time-domain waveforms
    axes[0, 0].plot(np.real(clean_waveform[:500]), label='I', alpha=0.7)
    axes[0, 0].plot(np.imag(clean_waveform[:500]), label='Q', alpha=0.7)
    axes[0, 0].set_title('Clean OFDM Waveform\n(Time Domain)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(np.real(noisy_waveform[:500]), label='I', alpha=0.7, color='orange')
    axes[0, 1].plot(np.imag(noisy_waveform[:500]), label='Q', alpha=0.7, color='red')
    axes[0, 1].set_title(f'Noisy OFDM Waveform\n(SNR = {snr_db} dB)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    if MODEL_AVAILABLE:
        axes[0, 2].plot(np.real(denoised_waveform[:500]), label='I', alpha=0.7, color='green')
        axes[0, 2].plot(np.imag(denoised_waveform[:500]), label='Q', alpha=0.7, color='cyan')
        axes[0, 2].set_title('AI Denoised Waveform\n(Time Domain)', fontweight='bold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'AI Model\nNot Available', 
                       ha='center', va='center', fontsize=16)
        axes[0, 2].axis('off')
    
    # Row 2: QPSK Constellations (after OFDM demodulation)
    qpsk_clean, _ = transceiver.ofdm_demod.demodulate(clean_waveform)
    qpsk_noisy, _ = transceiver.ofdm_demod.demodulate(noisy_waveform)
    
    axes[1, 0].scatter(np.real(qpsk_clean), np.imag(qpsk_clean), 
                      alpha=0.3, s=20, c='blue')
    axes[1, 0].set_title('Clean QPSK Constellation\n(After OFDM Demod)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color='k', linewidth=0.5)
    axes[1, 0].axvline(0, color='k', linewidth=0.5)
    axes[1, 0].set_xlabel('In-Phase (I)')
    axes[1, 0].set_ylabel('Quadrature (Q)')
    
    axes[1, 1].scatter(np.real(qpsk_noisy), np.imag(qpsk_noisy), 
                      alpha=0.3, s=20, c='orange')
    axes[1, 1].set_title(f'Noisy QPSK Constellation\nBER = {ber_noisy*100:.2f}%', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='k', linewidth=0.5)
    axes[1, 1].axvline(0, color='k', linewidth=0.5)
    axes[1, 1].set_xlabel('In-Phase (I)')
    axes[1, 1].set_ylabel('Quadrature (Q)')
    
    if MODEL_AVAILABLE:
        qpsk_denoised, _ = transceiver.ofdm_demod.demodulate(denoised_waveform)
        axes[1, 2].scatter(np.real(qpsk_denoised), np.imag(qpsk_denoised), 
                          alpha=0.3, s=20, c='green')
        axes[1, 2].set_title(f'AI Denoised Constellation\nBER = {ber_denoised*100:.2f}%', fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(0, color='k', linewidth=0.5)
        axes[1, 2].axvline(0, color='k', linewidth=0.5)
        axes[1, 2].set_xlabel('In-Phase (I)')
        axes[1, 2].set_ylabel('Quadrature (Q)')
    else:
        axes[1, 2].axis('off')
    
    # Row 3: Image Comparison
    axes[2, 0].imshow(test_image)
    axes[2, 0].set_title('Original Image\n(32x32 RGB)', fontweight='bold')
    axes[2, 0].axis('off')
    
    if noisy_image is not None:
        axes[2, 1].imshow(noisy_image)
        axes[2, 1].set_title(f'Noisy Decoded Image\nBER = {ber_noisy*100:.2f}%', fontweight='bold')
    else:
        axes[2, 1].text(0.5, 0.5, 'Decode\nFailed', ha='center', va='center', fontsize=16)
    axes[2, 1].axis('off')
    
    if MODEL_AVAILABLE and denoised_image is not None:
        axes[2, 2].imshow(denoised_image)
        axes[2, 2].set_title(f'AI Denoised Image\nBER = {ber_denoised*100:.2f}%', fontweight='bold')
    else:
        axes[2, 2].text(0.5, 0.5, 'AI Model\nNot Available' if not MODEL_AVAILABLE else 'Decode\nFailed', 
                       ha='center', va='center', fontsize=16)
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    # Save plot (don't show)
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'ofdm_ai_denoising_test.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close figure instead of showing
    print(f"📊 Plot saved: {plot_path}")
    
    # Summary
    print("\n" + "="*80)
    print(" "*30 + "SUMMARY")
    print("="*80)
    print(f"Original Data    : 32x32 RGB PNG ({len(message_bytes)} bytes)")
    print(f"Noisy Decoded    : {'SUCCESS' if noisy_image is not None else 'FAILED'}")
    if MODEL_AVAILABLE:
        print(f"AI Decoded       : {'SUCCESS' if denoised_image is not None else 'FAILED'}")
    print(f"\nBER (Control)    : {ber_noisy*100:.2f}%")
    if MODEL_AVAILABLE:
        print(f"BER (AI)         : {ber_denoised*100:.2f}%")
        improvement = ((ber_noisy - ber_denoised) / ber_noisy * 100) if ber_noisy > 0 else 0
        print(f"Improvement      : {improvement:.2f}%")
    print(f"\n📊 Results saved : output/ofdm_ai_denoising_test.png")
    print("="*80)


if __name__ == "__main__":
    run_test()
