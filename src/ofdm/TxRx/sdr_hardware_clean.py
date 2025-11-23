"""
================================================================================
SDR HARDWARE LIBRARY - PLUTO TX / RTL-SDR RX (CLEAN VERSION)
================================================================================
Clean transmission/reception pipeline using OFDM modulation.
No AI - just reliable Pluto → RTL-SDR wireless communication.

Hardware:
- TX: Adalm Pluto (via pyadi-iio)
- RX: RTL-SDR (via pyrtlsdr)

Features:
- OFDM Modulation (FFT=64, CP=16)
- Packet-based transmission with CRC32
- Real-time BER calculation
- Constellation visualization
- Message/file transfer

Usage:
    from sdr_hardware_clean import PlutoTX, RTLSDR_RX, run_loopback
    
    # Simple loopback test
    run_loopback("Hello from OFDM!", duration=3)
    
    # Manual control
    tx = PlutoTX()
    tx.transmit_message("Test message")
    
    rx = RTLSDR_RX()
    rx.receive_and_decode(duration=5)
================================================================================
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import threading

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
    from src.ofdm.core import OFDMTransceiver, OFDMParams, add_awgn_noise
    OFDM_AVAILABLE = True
except ImportError:
    OFDM_AVAILABLE = False
    print("⚠️  OFDM pipeline not available - check src/ofdm/core/")


# --- SDR CONFIGURATION ---
SDR_CONFIG = {
    'CENTER_FREQ': 915e6,        # 915 MHz ISM band
    'SAMPLE_RATE': 2e6,          # 2 MSPS
    'BANDWIDTH': 2e6,
    'TX_GAIN': -10,              # Pluto TX gain (dB)
    'RX_GAIN': 'auto',           # RTL-SDR auto gain
    'PLUTO_IP': "ip:192.168.2.1",
    'TX_BUFFER_SIZE': 65536,     # Minimum for GNU Radio compatibility
    'IQ_SCALE': 0.8,             # Prevent clipping (0.0-1.0)
}


class SignalUtils:
    """Helper utilities for signal analysis and visualization."""
    
    @staticmethod
    def calculate_ber(tx_bits, rx_bits):
        """Calculate Bit Error Rate."""
        min_len = min(len(tx_bits), len(rx_bits))
        if min_len == 0:
            return 1.0
        
        errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
        ber = errors / min_len
        return ber
    
    @staticmethod
    def plot_constellation(symbols, title="Constellation", expected_points=None):
        """Plot constellation diagram."""
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
    
    @staticmethod
    def plot_waveform(waveform, title="IQ Waveform", sample_rate=SDR_CONFIG['SAMPLE_RATE']):
        """Plot time and frequency domain."""
        plt.figure(figsize=(12, 8))
        
        # Time domain
        plt.subplot(2, 1, 1)
        plt.title(f"{title} - Time Domain")
        plt.plot(np.real(waveform[:1000]), label='I', alpha=0.7)
        plt.plot(np.imag(waveform[:1000]), label='Q', alpha=0.7)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        
        # Frequency domain
        plt.subplot(2, 1, 2)
        plt.title(f"{title} - Frequency Domain")
        plt.psd(waveform, NFFT=1024, Fs=sample_rate/1e6)
        plt.xlabel("Frequency (MHz)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class PlutoTX:
    """Adalm Pluto Transmitter with OFDM modulation."""
    
    def __init__(self, ip=SDR_CONFIG['PLUTO_IP']):
        self.ip = ip
        self.sdr = None
        self.is_connected = False
        
        if not OFDM_AVAILABLE:
            print("❌ OFDM pipeline not available!")
            return
        
        self.ofdm = OFDMTransceiver()
    
    def connect(self):
        """Connect to Pluto."""
        if adi is None:
            print("❌ pyadi-iio not installed!")
            return False
        
        print(f"🔌 Connecting to Pluto at {self.ip}...")
        try:
            self.sdr = adi.Pluto(self.ip)
            self.is_connected = True
            print("✅ Pluto connected")
            return True
        except Exception as e:
            print(f"❌ Pluto connection failed: {e}")
            return False
    
    def configure(self, center_freq=None, sample_rate=None, gain=None):
        """Configure TX parameters."""
        if not self.is_connected:
            if not self.connect():
                return False
        
        freq = center_freq or SDR_CONFIG['CENTER_FREQ']
        rate = sample_rate or SDR_CONFIG['SAMPLE_RATE']
        tx_gain = gain if gain is not None else SDR_CONFIG['TX_GAIN']
        
        try:
            self.sdr.tx_lo = int(freq)
            self.sdr.sample_rate = int(rate)
            self.sdr.tx_hardwaregain_chan0 = int(tx_gain)
            self.sdr.tx_cyclic_buffer = True
            
            print(f"⚙️  Pluto configured:")
            print(f"   Freq: {freq/1e6:.1f} MHz")
            print(f"   Rate: {rate/1e6:.1f} MSPS")
            print(f"   Gain: {tx_gain} dB")
            return True
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            return False
    
    def transmit_message(self, message, plot=True):
        """Transmit text message using OFDM."""
        if not self.is_connected:
            print("❌ Not connected. Call connect() first.")
            return
        
        print(f"\n📝 Message: '{message}'")
        
        # Generate OFDM waveform
        waveform, metadata = self.ofdm.transmit(message.encode('utf-8'))
        
        print(f"📊 OFDM Stats:")
        print(f"   Payload: {metadata['payload_bytes']} bytes")
        print(f"   Symbols: {metadata['num_symbols']}")
        print(f"   Samples: {len(waveform):,}")
        
        # Pad to minimum buffer size
        if len(waveform) < SDR_CONFIG['TX_BUFFER_SIZE']:
            pad_len = SDR_CONFIG['TX_BUFFER_SIZE'] - len(waveform)
            waveform = np.pad(waveform, (0, pad_len), mode='constant')
            print(f"   Padded to {len(waveform):,} samples (GNU Radio compat)")
        
        # Scale waveform
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform * SDR_CONFIG['IQ_SCALE'] / max_val
        
        print(f"   Power: {np.mean(np.abs(waveform)**2):.3f}")
        print(f"   Peak: {np.max(np.abs(waveform)):.3f}")
        
        if plot:
            SignalUtils.plot_waveform(waveform, "TX Waveform")
        
        # Transmit
        print(f"\n📡 Transmitting (cyclic)... Press Ctrl+C to stop")
        try:
            self.sdr.tx(waveform.astype(np.complex64))
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopped")
        finally:
            self.stop()
    
    def transmit_waveform(self, waveform, plot=False):
        """Transmit pre-generated waveform (non-blocking option for loopback)."""
        if not self.is_connected:
            print("❌ Not connected")
            return False
        
        # Pad and scale
        if len(waveform) < SDR_CONFIG['TX_BUFFER_SIZE']:
            pad_len = SDR_CONFIG['TX_BUFFER_SIZE'] - len(waveform)
            waveform = np.pad(waveform, (0, pad_len), mode='constant')
        
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform * SDR_CONFIG['IQ_SCALE'] / max_val
        
        if plot:
            SignalUtils.plot_waveform(waveform, "TX Waveform")
        
        print(f"📡 Starting transmission ({len(waveform):,} samples)...")
        self.sdr.tx(waveform.astype(np.complex64))
        return True
    
    def stop(self):
        """Stop transmission."""
        if self.sdr:
            self.sdr.tx_destroy_buffer()
            del self.sdr
            self.sdr = None
            self.is_connected = False
        print("✅ Pluto disconnected")


class RTLSDR_RX:
    """RTL-SDR Receiver with OFDM demodulation."""
    
    def __init__(self):
        self.sdr = None
        self.is_connected = False
        
        if not OFDM_AVAILABLE:
            print("❌ OFDM pipeline not available!")
            return
        
        self.ofdm = OFDMTransceiver()
    
    def connect(self):
        """Check RTL-SDR availability."""
        if RtlSdr is None:
            print("❌ pyrtlsdr not installed!")
            return False
        
        print("🔌 Checking RTL-SDR...")
        try:
            temp = RtlSdr()
            temp.close()
            self.is_connected = True
            print("✅ RTL-SDR detected")
            return True
        except Exception as e:
            print(f"❌ RTL-SDR not found: {e}")
            return False
    
    def configure(self, center_freq=None, sample_rate=None, gain=None):
        """Configure RX parameters."""
        freq = center_freq or SDR_CONFIG['CENTER_FREQ']
        rate = sample_rate or SDR_CONFIG['SAMPLE_RATE']
        rx_gain = gain or SDR_CONFIG['RX_GAIN']
        
        try:
            self.sdr = RtlSdr()
            self.sdr.center_freq = int(freq)
            self.sdr.sample_rate = int(rate)
            self.sdr.gain = rx_gain
            
            print(f"⚙️  RTL-SDR configured:")
            print(f"   Freq: {freq/1e6:.1f} MHz")
            print(f"   Rate: {rate/1e6:.1f} MSPS")
            print(f"   Gain: {rx_gain}")
            return True
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            if self.sdr:
                self.sdr.close()
                self.sdr = None
            return False
    
    def receive(self, duration=5, plot=True):
        """Capture IQ samples."""
        if not self.sdr:
            print("❌ Not configured. Call configure() first.")
            return None
        
        num_samples = int(self.sdr.sample_rate * duration)
        print(f"\n📥 Capturing {duration}s ({num_samples:,} samples)...")
        
        try:
            samples = self.sdr.read_samples(num_samples)
            print(f"✅ Captured {len(samples):,} samples")
            print(f"   Power: {np.mean(np.abs(samples)**2):.3f}")
            print(f"   Peak: {np.max(np.abs(samples)):.3f}")
            
            if plot:
                SignalUtils.plot_waveform(samples, "RX Waveform", self.sdr.sample_rate)
            
            return samples.astype(np.complex64)
        except Exception as e:
            print(f"❌ Capture failed: {e}")
            return None
    
    def receive_and_decode(self, duration=5, plot=True):
        """Receive and decode OFDM packets."""
        samples = self.receive(duration=duration, plot=False)
        
        if samples is None:
            return None
        
        print(f"\n🔍 Decoding OFDM...")
        
        try:
            decoded_data, stats = self.ofdm.receive(samples)
            
            print(f"\n📊 Reception Stats:")
            print(f"   Packets detected: {stats['packets_detected']}")
            print(f"   Packets valid: {stats['packets_decoded']}")
            print(f"   Packet Error Rate: {stats['packet_error_rate']*100:.1f}%")
            
            if stats['packets_decoded'] > 0:
                print(f"   Total bytes: {stats['total_bytes']}")
                
                # Try to decode as text
                try:
                    message = decoded_data.decode('utf-8')
                    print(f"\n✅ Decoded Message:")
                    print(f"   '{message}'")
                    
                    if plot:
                        # Plot constellation of received symbols
                        all_symbols = []
                        for pkt in stats.get('decoded_packets', []):
                            if 'equalized_symbols' in pkt:
                                all_symbols.extend(pkt['equalized_symbols'])
                        
                        if all_symbols:
                            # QPSK reference points
                            qpsk_ref = np.array([1+1j, 1-1j, -1+1j, -1-1j])
                            SignalUtils.plot_constellation(
                                np.array(all_symbols), 
                                "Received Constellation",
                                qpsk_ref
                            )
                    
                    return message
                except UnicodeDecodeError:
                    print(f"\n⚠️  Binary data ({len(decoded_data)} bytes)")
                    return decoded_data
            else:
                print("\n❌ No valid packets decoded")
                return None
                
        except Exception as e:
            print(f"❌ Decoding failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def close(self):
        """Close RTL-SDR."""
        if self.sdr:
            self.sdr.close()
            self.sdr = None
        print("✅ RTL-SDR closed")


def run_loopback(message="Hello OFDM!", duration=3, add_noise=False, snr_db=10):
    """
    Run complete loopback test: Pluto TX → RTL-SDR RX
    
    Args:
        message: Text to transmit
        duration: RX capture duration (seconds)
        add_noise: Add simulated noise for testing
        snr_db: SNR level if add_noise=True
    """
    print("="*80)
    print(" "*20 + "OFDM LOOPBACK TEST")
    print("="*80)
    
    # Initialize TX
    tx = PlutoTX()
    if not tx.connect():
        return False
    
    if not tx.configure():
        return False
    
    # Generate waveform
    print(f"\n📝 Message: '{message}'")
    waveform, metadata = tx.ofdm.transmit(message.encode('utf-8'))
    
    print(f"📊 OFDM Stats:")
    print(f"   Payload: {metadata['payload_bytes']} bytes")
    print(f"   Symbols: {metadata['num_symbols']}")
    print(f"   Samples: {len(waveform):,}")
    
    # Add noise if requested (for testing without RTL-SDR)
    if add_noise:
        print(f"\n🔊 Adding AWGN noise (SNR={snr_db} dB)")
        waveform = add_awgn_noise(waveform, snr_db)
    
    # Start TX in background
    print(f"\n🚀 Starting background transmission...")
    tx_thread = threading.Thread(
        target=tx.transmit_waveform,
        args=(waveform,),
        kwargs={'plot': False},
        daemon=True
    )
    tx_thread.start()
    time.sleep(2)  # Let TX stabilize
    
    # Initialize RX
    print(f"\n" + "-"*80)
    rx = RTLSDR_RX()
    if not rx.connect():
        tx.stop()
        return False
    
    if not rx.configure():
        tx.stop()
        return False
    
    # Receive and decode
    decoded = rx.receive_and_decode(duration=duration, plot=True)
    
    # Cleanup
    rx.close()
    tx.stop()
    
    # Results
    print(f"\n" + "="*80)
    if decoded:
        if isinstance(decoded, str):
            match = (decoded.strip() == message.strip())
            print(f"✅ SUCCESS: Message {'MATCH' if match else 'MISMATCH'}")
            print(f"   TX: '{message}'")
            print(f"   RX: '{decoded}'")
        else:
            print(f"✅ Received {len(decoded)} bytes of binary data")
    else:
        print(f"❌ FAILED: No data decoded")
    print("="*80)
    
    return decoded is not None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OFDM SDR Test')
    parser.add_argument('--mode', choices=['tx', 'rx', 'loopback'], default='loopback')
    parser.add_argument('--message', type=str, default='Hello OFDM!')
    parser.add_argument('--duration', type=int, default=3)
    parser.add_argument('--freq', type=float, help='Center frequency (Hz)')
    parser.add_argument('--gain', type=int, help='TX gain (dB)')
    
    args = parser.parse_args()
    
    # Update config if specified
    if args.freq:
        SDR_CONFIG['CENTER_FREQ'] = args.freq
    if args.gain:
        SDR_CONFIG['TX_GAIN'] = args.gain
    
    if args.mode == 'loopback':
        run_loopback(args.message, args.duration)
    elif args.mode == 'tx':
        tx = PlutoTX()
        if tx.connect() and tx.configure():
            tx.transmit_message(args.message)
    elif args.mode == 'rx':
        rx = RTLSDR_RX()
        if rx.connect() and rx.configure():
            rx.receive_and_decode(args.duration)
