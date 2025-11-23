import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import threading
from PIL import Image

# Hardware imports
try:
    import adi
except ImportError:
    adi = None
    
try:
    from rtlsdr import RtlSdr
except ImportError:
    RtlSdr = None

# Import OFDM pipeline
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
try:
    from src.ofdm.core import OFDMTransceiver, OFDMParams
    OFDM_AVAILABLE = True
except ImportError:
    OFDM_AVAILABLE = False
    print("⚠️  OFDM pipeline not available")

"""
================================================================================
SDR HARDWARE LIBRARY
================================================================================
This library provides unified classes for controlling Adalm Pluto (Tx) and 
RTL-SDR (Rx) devices. It includes built-in support for:
- QPSK Modulation/Demodulation (Image Transfer)
- AI Denoising (using PyTorch models)
- Signal Visualization (Time & Frequency Domain)

Usage:
    from sdr_hardware import PlutoTransmitter, RTLSDRReceiver, SDR_PARAMS

    # Modify defaults if needed
    SDR_PARAMS['CENTER_FREQ'] = 915e6

    # Transmit Image
    tx = PlutoTransmitter()
    if tx.connect():
        tx.transmit_image("my_image.png")

    # Receive & Denoise
    rx = RTLSDRReceiver()
    if rx.connect():
        data = rx.receive(duration=5)
        clean_data = rx.denoise_signal(data, model_path="model.pth")
================================================================================
"""

# --- DEFAULT CONFIGURATION ---
SDR_PARAMS = {
    'CENTER_FREQ': 915e6,    # 915 MHz (ISM Band)
    'SAMPLE_RATE': 2e6,      # 2 MSPS (RTL-SDR compatible)
    'BANDWIDTH': 2e6,        # 2 MHz (match sample rate)
    'TX_GAIN': -10,          # dB (Pluto)
    'RX_GAIN': 'auto',       # RTL-SDR auto gain
    'PLUTO_IP': "ip:192.168.2.1",
    'TX_BUFFER_SIZE': 65536, # GNU Radio compatibility
    'IQ_SCALE': 0.8,         # Scale to prevent clipping (0.0-1.0)
}

class SignalUtils:
    """Helper methods for Signal Processing, Plotting, and Conversion."""
    
    @staticmethod
    def bytes_to_qpsk(data_bytes):
        """Convert raw bytes to QPSK symbols."""
        if len(data_bytes) > SDR_PARAMS['MAX_FILE_SIZE']:
            print(f"⚠️ Warning: Data size {len(data_bytes)} exceeds recommended max {SDR_PARAMS['MAX_FILE_SIZE']}")
        
        # Convert bytes to bits
        bits = np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))
        
        # Pad to even number of bits for QPSK
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
        
        # QPSK Mapping (2 bits per symbol)
        # 00 -> -1-1j, 01 -> -1+1j, 10 -> 1-1j, 11 -> 1+1j
        symbols = []
        for i in range(0, len(bits), 2):
            b0 = bits[i]
            b1 = bits[i+1]
            re = 1 if b0 else -1
            im = 1 if b1 else -1
            symbols.append(re + 1j*im)
            
        symbols = np.array(symbols, dtype=np.complex64)
        symbols *= SDR_PARAMS['QPSK_SCALE']  # Scale for Pluto DAC
        return symbols
    
    @staticmethod
    def qpsk_to_bytes(symbols):
        """Convert QPSK symbols back to bytes."""
        # Normalize if scaled
        symbols = symbols / SDR_PARAMS['QPSK_SCALE']
        
        bits = []
        for s in symbols:
            re = np.real(s)
            im = np.imag(s)
            b0 = 1 if re > 0 else 0
            b1 = 1 if im > 0 else 0
            bits.extend([b0, b1])
            
        bits = np.array(bits, dtype=np.uint8)
        
        # Trim to byte boundary
        if len(bits) % 8 != 0:
            bits = bits[:-(len(bits)%8)]
            
        bytes_data = np.packbits(bits)
        return bytes_data.tobytes()
    
    @staticmethod
    def file_to_qpsk(file_path):
        """Convert any file to QPSK symbols with metadata."""
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            return None, None
        
        print(f"📁 Processing file: {path.name}")
        
        try:
            # Read file
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Create metadata header (filename + size)
            filename_bytes = path.name.encode('utf-8')[:255]  # Limit filename
            header = len(filename_bytes).to_bytes(1, 'big') + filename_bytes
            header += len(file_data).to_bytes(4, 'big')
            
            # Combine header + data
            full_data = header + file_data
            
            symbols = SignalUtils.bytes_to_qpsk(full_data)
            
            metadata = {
                'filename': path.name,
                'size': len(file_data),
                'total_symbols': len(symbols)
            }
            
            print(f"   File Size: {len(file_data)} bytes")
            print(f"   Symbols: {len(symbols)}")
            
            return symbols, metadata
            
        except Exception as e:
            print(f"❌ File conversion failed: {e}")
            return None, None
    
    @staticmethod
    def qpsk_to_file(symbols, output_path='received_file'):
        """Convert QPSK symbols back to file."""
        try:
            data_bytes = SignalUtils.qpsk_to_bytes(symbols)
            
            # Parse header
            if len(data_bytes) < 6:
                print("❌ Insufficient data to parse header")
                return False
            
            filename_len = data_bytes[0]
            if filename_len == 0 or filename_len > 255:
                print("❌ Invalid filename length in header")
                return False
                
            filename = data_bytes[1:1+filename_len].decode('utf-8', errors='ignore')
            file_size = int.from_bytes(data_bytes[1+filename_len:5+filename_len], 'big')
            
            # Extract file data
            file_data = data_bytes[5+filename_len:5+filename_len+file_size]
            
            # Save
            output_file = Path(output_path) / filename if Path(output_path).is_dir() else Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'wb') as f:
                f.write(file_data)
            
            print(f"💾 Saved: {output_file} ({len(file_data)} bytes)")
            return True
            
        except Exception as e:
            print(f"❌ File reconstruction failed: {e}")
            return False
    
    @staticmethod
    def image_to_qpsk(image_path, size=(64, 64)):
        """Convert image to QPSK symbols (Legacy wrapper)."""
        return SignalUtils.file_to_qpsk(image_path)

    @staticmethod
    def qpsk_to_image(symbols, size):
        """Convert QPSK symbols back to image (Legacy wrapper)."""
        return SignalUtils.qpsk_to_file(symbols)

    @staticmethod
    def plot_signal(data, title="Signal", sample_rate=SDR_PARAMS['SAMPLE_RATE']):
        """Helper to plot IQ data (Time & Frequency domain)."""
        plt.figure(figsize=(10, 8))
        
        # Time Domain
        plt.subplot(2, 1, 1)
        plt.title(f"{title} - Time Domain")
        plt.plot(np.real(data[:1000]), label='I (Real)', alpha=0.7)
        plt.plot(np.imag(data[:1000]), label='Q (Imag)', alpha=0.7)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Frequency Domain
        plt.subplot(2, 1, 2)
        plt.title(f"{title} - Frequency Domain")
        plt.psd(data, NFFT=1024, Fs=sample_rate/1e6)
        plt.xlabel("Frequency (MHz)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_comparison(original, denoised, title="Comparison"):
        """Plot comparison between Original (Noisy) and Denoised signals."""
        plt.figure(figsize=(12, 8))
        
        # Calculate dynamic limits based on actual signal amplitude
        max_val = max(np.abs(original).max(), np.abs(denoised).max()) * 1.2
        
        # 1. Waveforms
        plt.subplot(2, 2, 1)
        plt.title("Received (Noisy)")
        plt.plot(np.real(original[:500]), label='I', alpha=0.7, color='orange')
        plt.plot(np.imag(original[:500]), label='Q', alpha=0.7, color='red')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.title("Denoised (Clean)")
        plt.plot(np.real(denoised[:500]), label='I', alpha=0.7, color='blue')
        plt.plot(np.imag(denoised[:500]), label='Q', alpha=0.7, color='cyan')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Constellations
        plt.subplot(2, 2, 3)
        plt.title("Constellation (Noisy)")
        plt.scatter(np.real(original[:2000]), np.imag(original[:2000]), alpha=0.05, s=1, c='orange')
        plt.xlim(-max_val, max_val)
        plt.ylim(-max_val, max_val)
        plt.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        plt.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('I (Real)')
        plt.ylabel('Q (Imag)')
        
        plt.subplot(2, 2, 4)
        plt.title("Constellation (Denoised)")
        plt.scatter(np.real(denoised[:2000]), np.imag(denoised[:2000]), alpha=0.05, s=1, c='blue')
        plt.xlim(-max_val, max_val)
        plt.ylim(-max_val, max_val)
        plt.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        plt.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('I (Real)')
        plt.ylabel('Q (Imag)')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def calculate_ber(tx_bits, rx_bits):
        """Calculate Bit Error Rate between transmitted and received bits."""
        min_len = min(len(tx_bits), len(rx_bits))
        if min_len == 0:
            return 1.0
        
        errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
        ber = errors / min_len
        return ber
    
    @staticmethod
    def plot_constellation(symbols, title="Constellation", expected_points=None):
        """Plot constellation diagram with optional reference points."""
        plt.figure(figsize=(8, 8))
        plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.3, s=10, label='Received')
        
        if expected_points is not None:
            plt.scatter(np.real(expected_points), np.imag(expected_points), 
                       c='red', s=200, marker='x', linewidths=3, label='Expected')
        
        plt.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        plt.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('I (Real)')
        plt.ylabel('Q (Imag)')
        plt.title(title)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


class PlutoTransmitter:
    """
    Class to handle Adalm Pluto Transmitter operations.
    """
    def __init__(self, ip=SDR_PARAMS['PLUTO_IP']):
        self.ip = ip
        self.sdr = None
        self.is_connected = False

    def connect(self):
        """Check if Adalm Pluto is connected and initialize it."""
        if adi is None:
            print("❌ pyadi-iio not installed. Cannot use Pluto.")
            return False
            
        print(f"🔌 Connecting to Adalm Pluto at {self.ip}...")
        try:
            self.sdr = adi.Pluto(self.ip)
            self.is_connected = True
            print(f"✅ Connected to Pluto.")
            return True
        except Exception as e:
            print(f"❌ Could not connect to Pluto: {e}")
            self.is_connected = False
            return False

    def configure(self, center_freq=SDR_PARAMS['CENTER_FREQ'], sample_rate=SDR_PARAMS['SAMPLE_RATE'], gain=SDR_PARAMS['TX_GAIN']):
        """Configure SDR parameters."""
        if not self.is_connected:
            if not self.connect():
                return

        try:
            self.sdr.tx_lo = int(center_freq)
            self.sdr.tx_cyclic_buffer = True
            self.sdr.tx_hardwaregain_chan0 = int(gain)
            self.sdr.sample_rate = int(sample_rate)
            
            print(f"⚙️ Pluto Configured:")
            print(f"   Freq: {center_freq/1e6} MHz")
            print(f"   Rate: {sample_rate/1e6} MSPS")
            print(f"   Gain: {gain} dB")
        except Exception as e:
            print(f"❌ Configuration failed: {e}")

    def load_from_file(self, filepath):
        """Load IQ data from a file."""
        path = Path(filepath)
        if not path.exists():
            print(f"❌ File not found: {filepath}")
            return None
        
        try:
            data = np.fromfile(filepath, dtype=np.complex64)
            print(f"📂 Loaded {len(data)} samples from {path.name}")
            return data
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None

    def generate_test_signal(self, n_samples=1024):
        """Generate a random QPSK signal for testing."""
        data = (np.random.randint(0, 2, n_samples) * 2 - 1) + \
               1j * (np.random.randint(0, 2, n_samples) * 2 - 1)
        data = data * SDR_PARAMS['QPSK_SCALE']  # Scale for Pluto
        return data.astype(np.complex64)
    
    def transmit_file(self, file_path):
        """Convenience method to transmit any file (images, text, audio, binary)."""
        data, metadata = SignalUtils.file_to_qpsk(file_path)
        if data is not None:
            print(f"📡 Transmitting {metadata['filename']}...")
            self.transmit(data, plot=True)

    def transmit(self, data, plot=True):
        """Start continuous transmission of the provided data."""
        if not self.is_connected:
            print("❌ Pluto not connected.")
            return

        if data is None or len(data) == 0:
            print("❌ No data to transmit.")
            return

        if plot:
            print("📊 Plotting signal before transmission...")
            SignalUtils.plot_signal(data, title="Transmitted Signal")

        print(f"📡 Transmitting {len(data)} samples (Cyclic)... Press Ctrl+C to stop.")
        try:
            self.sdr.tx(data)
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ Stopping transmission...")
        except Exception as e:
            print(f"❌ Transmission error: {e}")
        finally:
            self.stop()

    def transmit_image(self, image_path):
        """Convenience method to transmit an image (Legacy alias for transmit_file)."""
        self.transmit_file(image_path)

    def transmit_stream(self, generator_func, chunk_size=1024):
        """
        Transmit a continuous stream of data generated by generator_func.
        Note: Pluto's cyclic buffer is efficient, but for true streaming 
        we need to disable cyclic buffer and push buffers continuously.
        """
        if not self.is_connected:
            print("❌ Pluto not connected.")
            return

        print("🌊 Starting Live Stream Transmission...")
        try:
            self.sdr.tx_cyclic_buffer = False # Disable cyclic for streaming
            
            while True:
                data = generator_func(chunk_size)
                self.sdr.tx(data)
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopping stream...")
        except Exception as e:
            print(f"❌ Stream error: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop transmission and release resources."""
        if self.sdr:
            del self.sdr
            self.sdr = None
            self.is_connected = False
        print("✅ Pluto disconnected.")


class RTLSDRReceiver:
    """
    Class to handle RTL-SDR Receiver operations.
    """
    def __init__(self):
        self.sdr = None
        self.is_connected = False

    def connect(self):
        """Check if RTL-SDR is connected."""
        if RtlSdr is None:
            print("❌ pyrtlsdr not installed. Cannot use RTL-SDR.")
            return False

        print("🔌 Checking for RTL-SDR...")
        try:
            # Just open and close to check presence
            temp_sdr = RtlSdr()
            temp_sdr.close()
            self.is_connected = True
            print("✅ RTL-SDR detected.")
            return True
        except Exception as e:
            print(f"❌ RTL-SDR not found: {e}")
            self.is_connected = False
            return False

    def configure(self, center_freq=SDR_PARAMS['CENTER_FREQ'], sample_rate=SDR_PARAMS['SAMPLE_RATE'], gain=SDR_PARAMS['RX_GAIN']):
        """Configure Receiver parameters."""
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = int(sample_rate)
            self.sdr.center_freq = int(center_freq)
            self.sdr.gain = gain
            
            print(f"⚙️ RTL-SDR Configured:")
            print(f"   Freq: {center_freq/1e6} MHz")
            print(f"   Rate: {sample_rate/1e6} MSPS")
            print(f"   Gain: {gain}")
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            if self.sdr:
                self.sdr.close()
                self.sdr = None

    def receive(self, duration=5, chunk_size=256*1024, plot=True):
        """
        Capture IQ data for a specific duration.
        Returns numpy array of complex64 samples.
        """
        if not self.sdr:
            print("❌ SDR not configured. Call configure() first.")
            return None

        num_samples = int(self.sdr.sample_rate * duration)
        print(f"📥 Capturing {duration}s ({num_samples} samples)...")
        
        samples = []
        remaining = num_samples
        
        try:
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                chunk = self.sdr.read_samples(to_read)
                samples.append(chunk)
                remaining -= len(chunk)
            
            full_data = np.concatenate(samples)
            print(f"✅ Capture complete: {len(full_data)} samples.")
            
            if plot:
                print("📊 Plotting received signal...")
                SignalUtils.plot_signal(full_data, title="Received Signal", sample_rate=self.sdr.sample_rate)
                
            return full_data.astype(np.complex64)
            
        except Exception as e:
            print(f"❌ Error during capture: {e}")
            return None

    def receive_and_process(self, duration=5, denoise=False, model_path=SDR_PARAMS['MODEL_PATH'], save_file=None):
        """
        High-level method to receive, optionally denoise, plot results, and save file.
        """
        data = self.receive(duration=duration, plot=False) # Don't plot yet
        
        if data is None:
            return None
            
        if denoise:
            clean_data = SignalUtils.denoise_signal(data, model_path)
            SignalUtils.plot_comparison(data, clean_data, title="Denoising Results")
            
            if save_file:
                SignalUtils.qpsk_to_file(clean_data, save_file)
            
            return clean_data
        else:
            SignalUtils.plot_signal(data, title="Received Signal")
            
            if save_file:
                SignalUtils.qpsk_to_file(data, save_file)
            
            return data

    def receive_stream(self, callback, chunk_size=8192):
        """
        Continuously receive data and pass it to a callback function.
        callback(data_chunk)
        """
        if not self.sdr:
            print("❌ SDR not configured.")
            return

        print("🌊 Starting Live Stream Reception... (Ctrl+C to stop)")
        try:
            while True:
                chunk = self.sdr.read_samples(chunk_size)
                if chunk is not None and len(chunk) > 0:
                    callback(chunk)
        except KeyboardInterrupt:
            print("\n⏹️ Stopping stream...")
        except Exception as e:
            print(f"❌ Stream error: {e}")
        finally:
            self.close()

    def save_to_file(self, data, filepath):
        """Save IQ data to file."""
        try:
            data.tofile(filepath)
            print(f"💾 Saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving file: {e}")

    def close(self):
        """Close the SDR connection."""
        if self.sdr:
            self.sdr.close()
            self.sdr = None
        print("✅ RTL-SDR closed.")
