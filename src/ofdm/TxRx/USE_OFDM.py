import sys
import argparse
import time
from pathlib import Path

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
   python universal_sdr.py --mode loopback --type file --file test.png --denoise --save output

================================================================================
"""

# --- DEFAULT CONFIGURATION (Modify as needed) ---
CONFIG = {
    'CENTER_FREQ': 915e6,
    'SAMPLE_RATE': 2e6,      # 2 MHz (RTL-SDR optimized)
    'TX_GAIN': -10,
    'RX_GAIN': 'auto',
    'DURATION': 5,
    'MODEL_PATH': 'saved_models/OFDM/unet1d_best.pth',
    'DEFAULT_FILE': 'src/ofdm/TxRx/Transcorn/testfile_img.png'
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
            if args.type == 'file':
                file_path = args.file if args.file else CONFIG['DEFAULT_FILE']
                print(f"📁 Transmitting file: {file_path}")
                tx_data, metadata = SignalUtils.file_to_qpsk(file_path)
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
            
            # Determine save path
            save_path = args.save if args.save else None
            
            # If denoising is requested, use the high-level method
            data = rx.receive_and_process(duration=args.duration, denoise=args.denoise, 
                                         model_path=args.model, save_file=save_path)

    print("\n✅ Operation Complete.")

if __name__ == "__main__":
    main()
