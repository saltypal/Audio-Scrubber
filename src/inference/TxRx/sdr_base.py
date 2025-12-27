"""
================================================================================
SDR BASE CLASSES - Hardware Abstraction
================================================================================
Base classes for Adalm Pluto (TX) and RTL-SDR (RX)
"""

import numpy as np
from abc import ABC, abstractmethod

try:
    import adi
    PLUTO_AVAILABLE = True
except ImportError:
    PLUTO_AVAILABLE = False
    adi = None

try:
    from rtlsdr import RtlSdr
    RTLSDR_AVAILABLE = True
except ImportError:
    RTLSDR_AVAILABLE = False
    RtlSdr = None


# Shared configuration dictionary (ISM band)
SDR_CONFIG = {
    'center_freq': 915e6,      # 915 MHz ISM band
    'sample_rate': 2e6,        # 2 MSPS
    'bandwidth': 2e6,          # 2 MHz
    'tx_gain': 0,              # 0 dB = Maximum TX power (No Attenuation)
    'rx_gain': 60,             # RTL-SDR max ~50-60 dB, Pluto max 76 dB - Crank up for better reception!
    'pluto_tx_ip': 'ip:192.168.2.1',  # TX Pluto IP
    'pluto_rx_ip': 'ip:192.168.3.1',  # RX Pluto IP (different device)
    'buffer_size': 65536,      # Minimum buffer size
}

# FM radio configuration (Broadcast FM band)
FM_CONFIG = {
    'center_freq': 105e6,      # 105 MHz (FM radio band 88-108 MHz)
    'sample_rate': 2e6,        # 2 MSPS
    'bandwidth': 2e6,          # 2 MHz
    'tx_gain': 0,              # 0 dB = Maximum TX power (No Attenuation)
    'rx_gain': 60,             # Crank up for better reception!
    'pluto_tx_ip': 'ip:192.168.2.1',  # TX Pluto IP
    'pluto_rx_ip': 'ip:192.168.3.1',  # RX Pluto IP (different device)
    'buffer_size': 65536,      # Minimum buffer size
}


class PlutoSDR:
    """Adalm Pluto SDR - Transmitter"""
    
    def __init__(self, ip=None):
        self.ip = ip or SDR_CONFIG['pluto_tx_ip']
        self.sdr = None
        self.connected = False
    
    def check_device(self):
        """Check if Pluto is available."""
        if not PLUTO_AVAILABLE:
            print("❌ pyadi-iio not installed!")
            return False
        
        print(f"🔌 Checking Pluto at {self.ip}...")
        try:
            self.sdr = adi.Pluto(self.ip)
            self.connected = True
            print("✅ Pluto detected")
            return True
        except Exception as e:
            print(f"❌ Pluto not found: {e}")
            self.connected = False
            return False
    
    def configure(self, freq=None, rate=None, gain=None, bandwidth=None):
        """Configure TX parameters."""
        if not self.connected:
            print("❌ Not connected. Call check_device() first.")
            return False
        
        try:
            self.sdr.tx_lo = int(freq or SDR_CONFIG['center_freq'])
            self.sdr.sample_rate = int(rate or SDR_CONFIG['sample_rate'])
            # Use maximum TX gain by default (-89 to 0 dB, 0 is max power)
            tx_gain = gain if gain is not None else SDR_CONFIG['tx_gain']
            self.sdr.tx_hardwaregain_chan0 = int(tx_gain)
            self.sdr.tx_rf_bandwidth = int(bandwidth or SDR_CONFIG['bandwidth'])
            self.sdr.tx_cyclic_buffer = True
            
            print(f"⚙️  Pluto configured:")
            print(f"   Freq: {self.sdr.tx_lo/1e6:.1f} MHz")
            print(f"   Rate: {self.sdr.sample_rate/1e6:.1f} MSPS")
            print(f"   Gain: {self.sdr.tx_hardwaregain_chan0} dB (max power)")
            return True
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            return False
    
    def transmit(self, waveform):
        """
        Transmit waveform (blocking, press Ctrl+C to stop).
        
        Args:
            waveform: Complex64 IQ samples
        """
        if not self.connected:
            print("❌ Not connected")
            return False
        
        # Ensure minimum buffer size
        if len(waveform) < SDR_CONFIG['buffer_size']:
            pad_len = SDR_CONFIG['buffer_size'] - len(waveform)
            waveform = np.pad(waveform, (0, pad_len), mode='constant')
        
        waveform = waveform.astype(np.complex64)
        
        print(f"📡 Transmitting {len(waveform):,} samples (Ctrl+C to stop)...")
        try:
            self.sdr.tx(waveform)
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopped")
            return True
        except Exception as e:
            print(f"❌ TX error: {e}")
            return False
    
    def transmit_background(self, waveform):
        """
        Start transmission in background (non-blocking).
        
        Args:
            waveform: Complex64 IQ samples
        """
        if not self.connected:
            print("❌ Not connected")
            return False
        
        # Ensure minimum buffer size
        if len(waveform) < SDR_CONFIG['buffer_size']:
            pad_len = SDR_CONFIG['buffer_size'] - len(waveform)
            waveform = np.pad(waveform, (0, pad_len), mode='constant')
        
        waveform = waveform.astype(np.complex64)
        
        try:
            self.sdr.tx(waveform)
            print(f"📡 Background TX started ({len(waveform):,} samples)")
            return True
        except Exception as e:
            print(f"❌ TX error: {e}")
            return False
    
    def stop(self):
        """Stop transmission and cleanup."""
        if self.sdr:
            try:
                self.sdr.tx_destroy_buffer()
            except:
                pass
            del self.sdr
            self.sdr = None
            self.connected = False
        print("✅ Pluto stopped")


class PlutoRX:
    """Adalm Pluto SDR - Receiver"""
    
    def __init__(self, ip=None):
        self.ip = ip or SDR_CONFIG['pluto_rx_ip']
        self.sdr = None
        self.connected = False
    
    def check_device(self):
        """Check if RX Pluto is available."""
        if not PLUTO_AVAILABLE:
            print("❌ pyadi-iio not installed!")
            return False
        
        print(f"🔌 Checking RX Pluto at {self.ip}...")
        try:
            self.sdr = adi.Pluto(self.ip)
            self.connected = True
            print("✅ RX Pluto detected")
            return True
        except Exception as e:
            print(f"❌ RX Pluto not found: {e}")
            self.connected = False
            return False
    
    def configure(self, freq=None, rate=None, gain=None, bandwidth=None):
        """Configure RX parameters."""
        if not self.connected:
            print("❌ Not connected. Call check_device() first.")
            return False
        
        try:
            self.sdr.rx_lo = int(freq or SDR_CONFIG['center_freq'])
            self.sdr.sample_rate = int(rate or SDR_CONFIG['sample_rate'])
            self.sdr.rx_hardwaregain_chan0 = int(gain if gain is not None else SDR_CONFIG['rx_gain'])
            self.sdr.rx_rf_bandwidth = int(bandwidth or SDR_CONFIG['bandwidth'])
            self.sdr.rx_buffer_size = SDR_CONFIG['buffer_size']
            
            print(f"⚙️  RX Pluto configured:")
            print(f"   Freq: {self.sdr.rx_lo/1e6:.1f} MHz")
            print(f"   Rate: {self.sdr.sample_rate/1e6:.1f} MSPS")
            print(f"   Gain: {self.sdr.rx_hardwaregain_chan0} dB")
            return True
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            return False
    
    def receive(self, duration=5):
        """
        Capture IQ samples.
        
        Args:
            duration: Capture duration in seconds
            
        Returns:
            Complex64 IQ samples
        """
        if not self.connected or not self.sdr:
            print("❌ Not configured. Call configure() first.")
            return None
        
        num_samples = int(self.sdr.sample_rate * duration)
        print(f"📥 Capturing {duration}s ({num_samples:,} samples)...")
        
        try:
            # Read samples from Pluto RX buffer
            samples = self.sdr.rx()
            
            # If we need more samples, keep reading
            if len(samples) < num_samples:
                all_samples = [samples]
                remaining = num_samples - len(samples)
                
                while remaining > 0:
                    chunk = self.sdr.rx()
                    all_samples.append(chunk)
                    remaining -= len(chunk)
                
                samples = np.concatenate(all_samples)[:num_samples]
            else:
                samples = samples[:num_samples]
            
            print(f"✅ Captured {len(samples):,} samples")
            print(f"   Power: {np.mean(np.abs(samples)**2):.3f}")
            print(f"   Peak: {np.max(np.abs(samples)):.3f}")
            
            return samples.astype(np.complex64)
        except Exception as e:
            print(f"❌ Capture failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def stop(self):
        """Stop and cleanup."""
        if self.sdr:
            try:
                self.sdr.rx_destroy_buffer()
            except:
                pass
            del self.sdr
            self.sdr = None
            self.connected = False
        print("✅ RX Pluto stopped")


class RTLSDR:
    """RTL-SDR - Receiver"""
    
    def __init__(self):
        self.sdr = None
        self.connected = False
    
    def check_device(self):
        """Check if RTL-SDR is available."""
        if not RTLSDR_AVAILABLE:
            print("❌ pyrtlsdr not installed!")
            return False
        
        print("🔌 Checking RTL-SDR...")
        try:
            temp = RtlSdr()
            temp.close()
            self.connected = True
            print("✅ RTL-SDR detected")
            return True
        except Exception as e:
            print(f"❌ RTL-SDR not found: {e}")
            self.connected = False
            return False
    
    def configure(self, freq=None, rate=None, gain=None):
        """Configure RX parameters."""
        try:
            self.sdr = RtlSdr()
            self.sdr.center_freq = int(freq or SDR_CONFIG['center_freq'])
            self.sdr.sample_rate = int(rate or SDR_CONFIG['sample_rate'])
            # Set gain to max (60 or 50 dB depending on stick)
            self.sdr.gain = gain if gain is not None else SDR_CONFIG['rx_gain']
            print(f"⚙️  RTL-SDR configured:")
            print(f"   Freq: {self.sdr.center_freq/1e6:.1f} MHz")
            print(f"   Rate: {self.sdr.sample_rate/1e6:.1f} MSPS")
            print(f"   Gain: {self.sdr.gain}")
            return True
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            if self.sdr:
                self.sdr.close()
                self.sdr = None
            return False
    
    def receive(self, duration=5):
        """
        Capture IQ samples.
        
        Args:
            duration: Capture duration in seconds
            
        Returns:
            Complex64 IQ samples
        """
        if not self.sdr:
            print("❌ Not configured. Call configure() first.")
            return None
        
        num_samples = int(self.sdr.sample_rate * duration)
        print(f"📥 Capturing {duration}s ({num_samples:,} samples)...")
        
        try:
            # Read in chunks to avoid buffer issues
            chunk_size = 256*1024
            samples = []
            remaining = num_samples

            # Ensure gain is set to max before capture (redundant but safe)
            self.sdr.gain = max(self.sdr.gain, 50)

            while remaining > 0:
                to_read = min(chunk_size, remaining)
                chunk = self.sdr.read_samples(to_read)
                samples.append(chunk)
                remaining -= len(chunk)
            
            samples = np.concatenate(samples)
            power = np.mean(np.abs(samples)**2)
            peak = np.max(np.abs(samples))
            
            print(f"✅ Captured {len(samples):,} samples")
            print(f"   Power: {power:.6f} ({10*np.log10(power + 1e-12):.2f} dB)")
            print(f"   Peak: {peak:.3f}")
            
            return samples.astype(np.complex64)
        except Exception as e:
            print(f"❌ Capture failed: {e}")
            return None
    
    def stop(self):
        """Stop and cleanup."""
        if self.sdr:
            self.sdr.close()
            self.sdr = None
        print("✅ RTL-SDR stopped")
