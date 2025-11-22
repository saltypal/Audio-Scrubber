import sys
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Add parent directory to path to allow importing sdr_hardware
sys.path.insert(0, str(Path(__file__).parent))
from sdr_hardware import PlutoTransmitter, RTLSDRReceiver, SDR_PARAMS, SignalUtils

"""
================================================================================
UNIVERSAL SDR CONTROLLER
================================================================================
This script provides a single entry point for all SDR operations:
- Transmitting (Random Data, Images, Text, Audio, or any Binary File)
- Receiving (Saving to file or Live Plotting)
- AI Denoising (Optional)
- Loopback Testing (Tx + Rx on same machine)

Usage Examples:

1. Transmit Random Data (Continuous Stream):
   python universal_sdr.py --mode tx --type random

2. Transmit ANY File (Image, Audio, Text, Binary):
   python universal_sdr.py --mode tx --type file --file my_document.pdf
   python universal_sdr.py --mode tx --type file --file song.mp3
   python universal_sdr.py --mode tx --type file --file data.bin

3. Receive Data (5 seconds) and Plot:
   python universal_sdr.py --mode rx --duration 5

4. Receive and Denoise using AI:
   python universal_sdr.py --mode rx --denoise

5. Receive and Save File:
   python universal_sdr.py --mode rx --save received_output

6. Loopback Test (Transmit File -> Receive -> Denoise -> Save):
================================================================================
"""

def visualize_images(original_path, noisy_path=None, denoised_path=None, save_path=None):
    """Display original, noisy (if available), and denoised (if available) images."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    if Path(original_path).exists():
        img = Image.open(original_path)
        axes[0].imshow(img)
        axes[0].set_title('Original (Before Tx)', fontweight='bold', fontsize=14)
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'Original\nNot Available', ha='center', va='center')
        axes[0].axis('off')
    
    # Noisy (Received)
    if noisy_path and Path(noisy_path).exists():
        img = Image.open(noisy_path)
        axes[1].imshow(img)
        axes[1].set_title('After Rx (Noisy)', fontweight='bold', fontsize=14)
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Received\nNot Available', ha='center', va='center')
        axes[1].axis('off')
    
    # Denoised
    if denoised_path and Path(denoised_path).exists():
        img = Image.open(denoised_path)
        axes[2].imshow(img)
        axes[2].set_title('After AI Denoise', fontweight='bold', fontsize=14)
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, 'Denoised\nNot Available', ha='center', va='center')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Comparison plot saved: {save_path}")
        plt.close()
    else:
        plt.show()

# --- DEFAULT CONFIGURATION (Modify as needed) ---
CONFIG = {
    'CENTER_FREQ': 915e6,
    'SAMPLE_RATE': 2e6,      # 2 MHz (RTL-SDR optimized)
    'TX_GAIN': -10,
    'RX_GAIN': 'auto',
    'DURATION': 5,
    'MODEL_PATH': 'saved_models/OFDM/unet1d_best.pth',
    'DEFAULT_FILE': 'src/ofdm/TxRx/Transcorn/testfile_img.png',
    'OUTPUT_DIR': 'src/ofdm/TxRx/Transcorn/output',
    'PLOT_DIR': 'src/ofdm/TxRx/Transcorn/plots'
}

def main():
    parser = argparse.ArgumentParser(description='Universal SDR Controller')
    
    # General Arguments
    parser.add_argument('--mode', choices=['tx', 'rx', 'loopback'], required=True, help='Operation mode')
    parser.add_argument('--type', choices=['random', 'file'], default='random', help='Data type to transmit (Tx/Loopback)')
    parser.add_argument('--file', type=str, help='Path to file to transmit or output directory for received file')
    
    # SDR Configuration
    parser.add_argument('--freq', type=float, default=CONFIG['CENTER_FREQ'], help='Center Frequency (Hz)')
    parser.add_argument('--rate', type=float, default=CONFIG['SAMPLE_RATE'], help='Sample Rate (Hz)')
    parser.add_argument('--gain', type=str, default=str(CONFIG['TX_GAIN']), help='Gain (dB). Tx default -10, Rx default auto')
    
    # Rx Specific
    parser.add_argument('--duration', type=int, default=CONFIG['DURATION'], help='Capture duration in seconds')
    parser.add_argument('--denoise', action='store_true', help='Apply AI Denoising to received signal')
    parser.add_argument('--model', type=str, default=CONFIG['MODEL_PATH'], help='Path to AI model')
    parser.add_argument('--save', type=str, help='Output file/directory for received data')
    
    args = parser.parse_args()

    # Update Global Params from Args
    SDR_PARAMS['CENTER_FREQ'] = args.freq
    SDR_PARAMS['SAMPLE_RATE'] = args.rate
    
    # Parse Gain
    tx_gain = int(float(args.gain)) if args.mode in ['tx', 'loopback'] else -10
    rx_gain = 'auto'
    if args.mode in ['rx', 'loopback']:
        try:
            rx_gain = float(args.gain)
        except:
            rx_gain = 'auto'

    print("=" * 60)
    print(f"      UNIVERSAL SDR CONTROLLER ({args.mode.upper()})")
    print("=" * 60)

    # --- TRANSMITTER LOGIC ---
    if args.mode in ['tx', 'loopback']:
        tx = PlutoTransmitter()
        if tx.connect():
            tx.configure(gain=tx_gain)
            
            # Prepare Data
            tx_data = None
            original_file = None
            if args.type == 'file':
                file_path = args.file if args.file else CONFIG['DEFAULT_FILE']
                original_file = file_path
                print(f"📁 Transmitting file: {file_path}")
                tx_data, metadata = SignalUtils.file_to_qpsk(file_path)
                
                # Save original signal plot before transmission
                print("\n📊 Saving signal plot before transmission...")
                plot_dir = Path(CONFIG['PLOT_DIR'])
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                plt.figure(figsize=(10, 8))
                plt.subplot(2, 1, 1)
                plt.title("Original Signal (Before Tx) - Time Domain")
                plt.plot(np.real(tx_data[:1000]), label='I (Real)', alpha=0.7)
                plt.plot(np.imag(tx_data[:1000]), label='Q (Imag)', alpha=0.7)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 1, 2)
                plt.title("Original Signal (Before Tx) - Frequency Domain")
                plt.psd(tx_data, NFFT=1024, Fs=SDR_PARAMS['SAMPLE_RATE']/1e6)
                plt.xlabel("Frequency (MHz)")
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                tx_plot_path = plot_dir / 'signal_before_tx.png'
                plt.savefig(tx_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   Saved: {tx_plot_path}")
            else:
                print("🎲 Generating Random QPSK Data...")
                tx_data = tx.generate_test_signal()

            if tx_data is not None:
                if args.mode == 'tx':
                    # Blocking Transmission
                    tx.transmit(tx_data)
                else:
                    # Loopback: Start Tx in background
                    import threading
                    print("🚀 Starting Background Transmission for Loopback...")
                    t = threading.Thread(target=tx.transmit, args=(tx_data,), kwargs={'plot':False}, daemon=True)
                    t.start()
                    time.sleep(2) # Wait for Tx to stabilize

    # --- RECEIVER LOGIC ---
    if args.mode in ['rx', 'loopback']:
        rx = RTLSDRReceiver()
        if rx.connect():
            rx.configure(gain=rx_gain)
            
            # Receive & Process
            print(f"📥 Receiving for {args.duration} seconds...")
            
            # Receive data
            data = rx.receive(duration=args.duration, plot=False)
            
            if data is not None:
                # Create output directories
                output_dir = Path(args.save) if args.save else Path(CONFIG['OUTPUT_DIR'])
                plot_dir = Path(CONFIG['PLOT_DIR'])
                output_dir.mkdir(parents=True, exist_ok=True)
                plot_dir.mkdir(parents=True, exist_ok=True)
                
                # Save received signal plot
                print("\n📊 Saving received signal plot...")
                plt.figure(figsize=(10, 8))
                plt.subplot(2, 1, 1)
                plt.title("Received Signal (After Rx) - Time Domain")
                plt.plot(np.real(data[:1000]), label='I (Real)', alpha=0.7)
                plt.plot(np.imag(data[:1000]), label='Q (Imag)', alpha=0.7)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 1, 2)
                plt.title("Received Signal (After Rx) - Frequency Domain")
                plt.psd(data, NFFT=1024, Fs=SDR_PARAMS['SAMPLE_RATE']/1e6)
                plt.xlabel("Frequency (MHz)")
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                rx_plot_path = plot_dir / 'signal_after_rx.png'
                plt.savefig(rx_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   Saved: {rx_plot_path}")
                
                if args.denoise:
                    # Save noisy version first
                    noisy_file = output_dir / 'received_noisy.png'
                    SignalUtils.qpsk_to_file(data, noisy_file)
                    print(f"\n💾 Saved noisy: {noisy_file}")
                    
                    # Denoise
                    print("\n🧠 Applying AI denoising...")
                    clean_data = SignalUtils.denoise_signal(data, args.model)
                    
                    # Save denoised version
                    denoised_file = output_dir / 'received_denoised.png'
                    SignalUtils.qpsk_to_file(clean_data, denoised_file)
                    print(f"💾 Saved denoised: {denoised_file}")
                    
                    # Save 3-image comparison: Original, Noisy, Denoised
                    comparison_plot = plot_dir / 'image_comparison.png'
                    if args.mode == 'loopback' and original_file:
                        print("\n🖼️  Saving comparison: Original → Noisy → Denoised")
                        visualize_images(original_file, noisy_file, denoised_file, save_path=comparison_plot)
                    else:
                        print("\n🖼️  Saving comparison: Noisy → Denoised")
                        visualize_images(noisy_file, noisy_file, denoised_file, save_path=comparison_plot)
                else:
                    # No denoising - just save received
                    output_file = output_dir / 'received.png'
                    SignalUtils.qpsk_to_file(data, output_file)
                    print(f"\n💾 Saved received: {output_file}")
                    
                    # Save comparison: Original vs Received
                    if args.mode == 'loopback' and original_file:
                        comparison_plot = plot_dir / 'image_comparison.png'
                        print("\n🖼️  Saving comparison: Original → Received")
                        visualize_images(original_file, output_file, None, save_path=comparison_plot)
            
            rx.close()

    print("\n✅ Operation Complete.")

if __name__ == "__main__":
    main()
