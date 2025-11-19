import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Hardware imports
try:
    import adi
except ImportError:
    adi = None
    
try:
    from rtlsdr import RtlSdr
except ImportError:
    RtlSdr = None

"""
Unified OFDM Transmitter/Receiver
Keeps all RF parameters synchronized between Tx and Rx
"""

class OFDMConfig:
    """Shared configuration for Tx and Rx"""
    CENTER_FREQ = 915e6    # 915 MHz (ISM Band)
    SAMPLE_RATE = 1e6      # 1 MSPS
    TX_GAIN = -10          # dB (Pluto)
    RX_GAIN = 'auto'       # RTL-SDR
    PLUTO_IP = "ip:192.168.2.1"
    
class PlutoTransmitter:
    def __init__(self, config=None):
        if adi is None:
            raise ImportError("pyadi-iio not installed. Run: pip install pyadi-iio")
        
        self.config = config or OFDMConfig()
        self.sdr = None
        
    def check_device(self):
        """Check if Adalm Pluto is connected"""
        print("Checking for Adalm Pluto...")
        try:
            self.sdr = adi.Pluto(self.config.PLUTO_IP)
            print(f"✅ Found Pluto at {self.config.PLUTO_IP}")
            return True
        except Exception as e:
            print(f"❌ Could not connect to Pluto: {e}")
            return False
    
    def configure(self):
        """Configure SDR for transmission"""
        if self.sdr is None:
            raise RuntimeError("Device not initialized. Call check_device() first.")
        
        self.sdr.tx_lo = int(self.config.CENTER_FREQ)
        self.sdr.tx_cyclic_buffer = True
        self.sdr.tx_hardwaregain_chan0 = int(self.config.TX_GAIN)
        self.sdr.sample_rate = int(self.config.SAMPLE_RATE)
        
        print(f"Configured:")
        print(f"  Frequency: {self.config.CENTER_FREQ/1e6} MHz")
        print(f"  Sample Rate: {self.config.SAMPLE_RATE/1e6} MSPS")
        print(f"  Gain: {self.config.TX_GAIN} dB")
    
    def generate_ofdm_signal(self, n_samples=1024):
        """Generate a simple QPSK OFDM-like signal"""
        data = (np.random.randint(0, 2, n_samples) * 2 - 1) + \
               1j * (np.random.randint(0, 2, n_samples) * 2 - 1)
        data = data * 2**14  # Scale for Pluto (14-bit DAC)
        return data
    
    def transmit(self, data=None):
        """Transmit signal continuously"""
        if data is None:
            print("Generating test signal...")
            data = self.generate_ofdm_signal()
        
        print("Transmitting... Press Ctrl+C to stop.")
        try:
            self.sdr.tx(data)
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping transmission...")
        finally:
            del self.sdr
            print("Done.")

class RTLSDRReceiver:
    def __init__(self, config=None):
        if RtlSdr is None:
            raise ImportError("pyrtlsdr not installed. Run: pip install pyrtlsdr")
        
        self.config = config or OFDMConfig()
        self.sdr = None
        
    def check_device(self):
        """Check if RTL-SDR is connected"""
        print("Checking for RTL-SDR...")
        try:
            self.sdr = RtlSdr()
            print(f"✅ Found RTL-SDR")
            self.sdr.close()
            self.sdr = None
            return True
        except Exception as e:
            print(f"❌ Could not connect to RTL-SDR: {e}")
            return False
    
    def configure(self, duration=5):
        """Configure SDR for reception"""
        self.sdr = RtlSdr()
        self.sdr.sample_rate = int(self.config.SAMPLE_RATE)
        self.sdr.center_freq = int(self.config.CENTER_FREQ)
        self.sdr.gain = self.config.RX_GAIN
        self.duration = duration
        
        print(f"Configured:")
        print(f"  Frequency: {self.config.CENTER_FREQ/1e6} MHz")
        print(f"  Sample Rate: {self.config.SAMPLE_RATE/1e6} MSPS")
        print(f"  Gain: {self.config.RX_GAIN}")
        print(f"  Duration: {duration} seconds")
    
    def receive(self, save_filename="captured_ofdm.iq", plot=True):
        """Capture IQ data"""
        num_samples = int(self.config.SAMPLE_RATE * self.duration)
        print(f"Capturing {num_samples} samples...")
        
        try:
            # Read in smaller chunks to avoid USB buffer issues
            samples = []
            chunk_size = 256 * 1024  # 256K samples per read
            remaining = num_samples
            
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                chunk = self.sdr.read_samples(to_read)
                samples.append(chunk)
                remaining -= len(chunk)
            
            samples = np.concatenate(samples)
            print("✅ Capture complete.")
            
            # Save
            print(f"Saving to {save_filename}...")
            samples.astype(np.complex64).tofile(save_filename)
            
            # Plot
            if plot:
                print("Plotting spectrum...")
                plt.figure()
                plt.psd(samples, NFFT=1024, 
                       Fs=self.config.SAMPLE_RATE/1e6, 
                       Fc=self.config.CENTER_FREQ/1e6)
                plt.title("Received Spectrum")
                plt.xlabel("Frequency (MHz)")
                plt.show()
                
            return samples
            
        except Exception as e:
            print(f"❌ Error during capture: {e}")
            return None
        finally:
            self.sdr.close()

def main_tx():
    """Transmitter Mode"""
    print("=" * 50)
    print("      OFDM TRANSMITTER (Adalm Pluto)")
    print("=" * 50)
    
    tx = PlutoTransmitter()
    
    if not tx.check_device():
        print("Pluto not detected. Exiting.")
        return
    
    resp = input("\nProceed with transmission? (y/n): ")
    if resp.lower() != 'y':
        print("Aborted.")
        return
    
    tx.configure()
    tx.transmit()

def main_rx():
    """Receiver Mode"""
    print("=" * 50)
    print("      OFDM RECEIVER (RTL-SDR)")
    print("=" * 50)
    
    rx = RTLSDRReceiver()
    
    if not rx.check_device():
        print("RTL-SDR not detected. Exiting.")
        return
    
    resp = input("\nProceed with reception? (y/n): ")
    if resp.lower() != 'y':
        print("Aborted.")
        return
    
    rx.configure(duration=5)
    rx.receive()

def main_txrx():
    """Run both Transmitter and Receiver together"""
    print("=" * 50)
    print("   OFDM TRANSMITTER & RECEIVER")
    print("=" * 50)
    
    # Initialize both
    tx = PlutoTransmitter()
    rx = RTLSDRReceiver()
    
    # Check devices
    print("\n[1/2] Checking Transmitter...")
    tx_ok = tx.check_device()
    
    print("\n[2/2] Checking Receiver...")
    rx_ok = rx.check_device()
    
    if not tx_ok and not rx_ok:
        print("\n❌ Neither device detected. Exiting.")
        return
    
    print("\n" + "=" * 50)
    print("Device Status:")
    print(f"  Pluto (TX): {'✅ Ready' if tx_ok else '❌ Not Found'}")
    print(f"  RTL-SDR (RX): {'✅ Ready' if rx_ok else '❌ Not Found'}")
    print("=" * 50)
    
    # Menu
    while True:
        print("\nOptions:")
        if tx_ok:
            print("  1. Start Transmission (Pluto)")
        if rx_ok:
            print("  2. Start Reception (RTL-SDR)")
        if tx_ok and rx_ok:
            print("  3. Transmit & Receive (Loopback Test)")
        print("  q. Quit")
        
        choice = input("\nSelect: ").strip().lower()
        
        if choice == 'q':
            print("Exiting.")
            break
        elif choice == '1' and tx_ok:
            tx.configure()
            tx.transmit()
        elif choice == '2' and rx_ok:
            duration = input("Duration (seconds, default 5): ").strip()
            duration = int(duration) if duration else 5
            rx.configure(duration=duration)
            rx.receive()
        elif choice == '3' and tx_ok and rx_ok:
            print("\n[LOOPBACK TEST]")
            print("1. Starting transmission in background...")
            tx.configure()
            
            # Start transmission in a thread
            import threading
            tx_thread = threading.Thread(target=tx.transmit, daemon=True)
            tx_thread.start()
            
            time.sleep(2)  # Let transmitter stabilize
            
            print("2. Starting reception...")
            rx.configure(duration=5)
            rx.receive()
            
            print("3. Loopback test complete.")
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OFDM Transmitter/Receiver')
    parser.add_argument('mode', nargs='?', choices=['tx', 'rx', 'both'], 
                       default='both', help='Mode: tx, rx, or both (default)')
    
    args = parser.parse_args()
    
    if args.mode == 'tx':
        main_tx()
    elif args.mode == 'rx':
        main_rx()
    else:
        main_txrx()
