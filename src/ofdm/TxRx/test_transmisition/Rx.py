import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Try importing pyrtlsdr
try:
    from rtlsdr import RtlSdr
except ImportError:
    print("Error: 'rtlsdr' library not found. Please install it using: pip install pyrtlsdr")
    sys.exit(1)

def check_rtlsdr():
    """Checks if RTL-SDR is connected."""
    print("Checking for RTL-SDR...")
    try:
        sdr = RtlSdr()
        print(f"Found RTL-SDR")
        sdr.close() # Close for now, reopen later
        return True
    except Exception as e:
        print(f"Could not connect to RTL-SDR: {e}")
        return False

def main():
    print("========================================")
    print("      OFDM RECEIVER (RTL-SDR)           ")
    print("========================================")
    
    # 1. Check Hardware
    if not check_rtlsdr():
        print("RTL-SDR not detected. Exiting.")
        return

    # 2. Configuration
    CENTER_FREQ = 915e6 # Must match Tx
    SAMPLE_RATE = 1e6   # Must match Tx
    GAIN = 'auto'
    DURATION = 5        # Seconds to record
    
    print(f"\nConfiguration:")
    print(f"  Frequency: {CENTER_FREQ/1e6} MHz")
    print(f"  Sample Rate: {SAMPLE_RATE/1e6} MSPS")
    print(f"  Gain: {GAIN}")
    print(f"  Duration: {DURATION} seconds")
    
    # 3. User Confirmation
    resp = input("\nProceed with reception? (y/n): ")
    if resp.lower() != 'y':
        print("Aborted.")
        return

    # 4. Capture
    sdr = RtlSdr()
    try:
        sdr.sample_rate = int(SAMPLE_RATE)
        sdr.center_freq = int(CENTER_FREQ)
        sdr.gain = GAIN
        
        num_samples = int(SAMPLE_RATE * DURATION)
        print(f"Capturing {num_samples} samples...")
        
        # Read in smaller chunks to avoid USB buffer issues
        samples_list = []
        chunk_size = 256 * 1024  # 256K samples per read
        remaining = num_samples
        
        while remaining > 0:
            to_read = min(chunk_size, remaining)
            chunk = sdr.read_samples(to_read)
            samples_list.append(chunk)
            remaining -= len(chunk)
        
        samples = np.concatenate(samples_list)
        print("Capture complete.")
        
    except Exception as e:
        print(f"Error during capture: {e}")
        return
    finally:
        sdr.close()

    # 5. Save/Visualize
    filename = "captured_ofdm.iq"
    print(f"Saving to {filename}...")
    # Save as complex64 (standard IQ format)
    samples.astype(np.complex64).tofile(filename)
    
    # Quick Plot
    print("Plotting spectrum...")
    plt.figure()
    plt.psd(samples, NFFT=1024, Fs=SAMPLE_RATE/1e6, Fc=CENTER_FREQ/1e6)
    plt.title("Received Spectrum")
    plt.xlabel("Frequency (MHz)")
    plt.show()
    
    print("Done.")

if __name__ == "__main__":
    main()
