import sys
import time
import numpy as np
import argparse

# Try importing ADI library for Pluto
try:
    import adi
except ImportError:
    print("Error: 'adi' library not found. Please install it using: pip install pyadi-iio")
    sys.exit(1)

def check_pluto():
    """Checks if Adalm Pluto is connected."""
    print("Checking for Adalm Pluto...")
    try:
        # Try default IP
        sdr = adi.Pluto("ip:192.168.2.1")
        print(f"Found Pluto at 192.168.2.1")
        return sdr
    except Exception as e:
        print(f"Could not connect to Pluto: {e}")
        return None

def main():
    print("========================================")
    print("      OFDM TRANSMITTER (Adalm Pluto)    ")
    print("========================================")
    
    # 1. Check Hardware
    sdr = check_pluto()
    if sdr is None:
        print("Pluto not detected. Exiting.")
        return

    # 2. Configuration
    CENTER_FREQ = 915e6 # 915 MHz (ISM Band)
    SAMPLE_RATE = 1e6   # 1 MSPS
    GAIN = -10          # dB
    
    print(f"\nConfiguration:")
    print(f"  Frequency: {CENTER_FREQ/1e6} MHz")
    print(f"  Sample Rate: {SAMPLE_RATE/1e6} MSPS")
    print(f"  Gain: {GAIN} dB")
    
    # 3. User Confirmation
    resp = input("\nProceed with transmission? (y/n): ")
    if resp.lower() != 'y':
        print("Aborted.")
        return

    # 4. Configure SDR
    sdr.tx_lo = int(CENTER_FREQ)
    sdr.tx_cyclic_buffer = True # Transmit continuously
    sdr.tx_hardwaregain_chan0 = int(GAIN)
    sdr.sample_rate = int(SAMPLE_RATE)
    
    # 5. Generate/Load Data
    # For now, let's generate a simple QPSK OFDM-like signal
    # Or load from file if provided
    print("Generating test signal...")
    N = 1024
    # Random QPSK
    data = (np.random.randint(0, 2, N) * 2 - 1) + 1j * (np.random.randint(0, 2, N) * 2 - 1)
    data = data * 2**14 # Scale for Pluto (14-bit DAC)
    
    # 6. Transmit
    print("Transmitting... Press Ctrl+C to stop.")
    try:
        sdr.tx(data) # Send data to buffer
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping transmission...")
    finally:
        # Cleanup if needed (adi handles it mostly)
        del sdr
        print("Done.")

if __name__ == "__main__":
    main()
