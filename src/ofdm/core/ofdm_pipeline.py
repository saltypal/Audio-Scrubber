"""
================================================================================
OFDM PIPELINE - CORRECT ARCHITECTURE
================================================================================
Pipeline Flow:
    TX: Text/Binary → QPSK Symbols → OFDM Modulation (IFFT+CP) → IQ Waveform → SDR
    RX: SDR → IQ Waveform → AI Denoise → OFDM Demodulation (FFT) → QPSK Decode → Text/Binary

Key Insight: AI works on TIME-DOMAIN OFDM WAVEFORM, NOT on QPSK symbols!
================================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import struct
import binascii

@dataclass
class OFDMParams:
    """
    OFDM System Parameters - MATCHED TO GNU RADIO
    
    These parameters match the GNU Radio OFDM Transmitter block:
    - FFT Length: 64
    - Cyclic Prefix Length: 16
    - Occupied Carriers: ((-4, -3, -2, -1, 1, 2, 3, 4),)
    - Pilot Carriers: ((-21, -7, 7, 21),)
    - Pilot Symbols: ((1, 1, 1, -1),)
    - Header/Payload Modulation: QPSK
    """
    fft_size: int = 64
    cp_len: int = 16
    sample_rate: float = 2e6  # 2 MHz
    
    # Subcarrier allocation (MATCHED TO GNU RADIO)
    data_carriers: np.ndarray = None  # Indices for data
    pilot_carriers: np.ndarray = None  # Indices for pilots
    pilot_values: np.ndarray = None  # Known pilot symbols
    
    # Power normalization
    target_power: float = 35.0
    
    def __post_init__(self):
        if self.data_carriers is None:
            # GNU Radio Occupied Carriers: ((-4, -3, -2, -1, 1, 2, 3, 4),)
            self.data_carriers = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
        if self.pilot_carriers is None:
            # GNU Radio Pilot Carriers: ((-21, -7, 7, 21),)
            self.pilot_carriers = np.array([-21, -7, 7, 21])
        if self.pilot_values is None:
            # GNU Radio Pilot Symbols: ((1, 1, 1, -1),)
            self.pilot_values = np.array([1, 1, 1, -1], dtype=np.complex64)
    
    @property
    def symbol_length(self):
        return self.fft_size + self.cp_len
    
    @property
    def num_data_carriers(self):
        return len(self.data_carriers)


class QPSKModulator:
    """QPSK Modulation/Demodulation"""
    
    @staticmethod
    def modulate(bits: np.ndarray) -> np.ndarray:
        """
        Convert bits to QPSK symbols
        00 -> -1-1j, 01 -> -1+1j, 10 -> 1-1j, 11 -> 1+1j
        """
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
        
        symbols = []
        for i in range(0, len(bits), 2):
            b0, b1 = bits[i], bits[i+1]
            re = 1 if b0 else -1
            im = 1 if b1 else -1
            symbols.append(re + 1j*im)
        
        return np.array(symbols, dtype=np.complex64) / np.sqrt(2)  # Normalize power
    
    @staticmethod
    def demodulate(symbols: np.ndarray) -> np.ndarray:
        """Convert QPSK symbols back to bits"""
        bits = []
        for s in symbols:
            bits.append(1 if np.real(s) > 0 else 0)
            bits.append(1 if np.imag(s) > 0 else 0)
        return np.array(bits, dtype=np.uint8)


class OFDMModulator:
    """OFDM Modulation Engine (IFFT + Cyclic Prefix)"""
    
    def __init__(self, params: OFDMParams):
        self.params = params
    
    def modulate(self, qpsk_symbols: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Convert QPSK symbols → OFDM time-domain waveform
        
        Args:
            qpsk_symbols: Complex QPSK symbols
            
        Returns:
            waveform: Time-domain IQ samples (complex64)
            metadata: Info about the transmission
        """
        # 1. Calculate number of OFDM symbols needed
        num_ofdm_symbols = int(np.ceil(len(qpsk_symbols) / self.params.num_data_carriers))
        total_capacity = num_ofdm_symbols * self.params.num_data_carriers
        
        # 2. Pad QPSK symbols to fill OFDM symbols
        if len(qpsk_symbols) < total_capacity:
            padding = np.zeros(total_capacity - len(qpsk_symbols), dtype=np.complex64)
            qpsk_symbols = np.concatenate([qpsk_symbols, padding])
        
        # 3. Generate OFDM symbols
        waveform = []
        for i in range(num_ofdm_symbols):
            # Extract chunk of QPSK symbols for this OFDM symbol
            chunk = qpsk_symbols[i * self.params.num_data_carriers : (i+1) * self.params.num_data_carriers]
            
            # Create frequency-domain OFDM symbol
            freq_domain = np.zeros(self.params.fft_size, dtype=np.complex64)
            
            # Map data carriers
            for idx, carrier_idx in enumerate(self.params.data_carriers):
                freq_domain[carrier_idx % self.params.fft_size] = chunk[idx]
            
            # Map pilot carriers
            for idx, carrier_idx in enumerate(self.params.pilot_carriers):
                freq_domain[carrier_idx % self.params.fft_size] = self.params.pilot_values[idx]
            
            # IFFT to time domain
            time_domain = np.fft.ifft(freq_domain)
            
            # Add Cyclic Prefix
            cp = time_domain[-self.params.cp_len:]
            ofdm_symbol = np.concatenate([cp, time_domain])
            
            waveform.extend(ofdm_symbol)
        
        waveform = np.array(waveform, dtype=np.complex64)
        
        # 4. Power Normalization
        current_power = np.mean(np.abs(waveform)**2)
        scale = np.sqrt(self.params.target_power / current_power) if current_power > 0 else 1.0
        waveform *= scale
        
        metadata = {
            'num_ofdm_symbols': num_ofdm_symbols,
            'num_qpsk_symbols': len(qpsk_symbols),
            'waveform_length': len(waveform),
            'power_before_scale': current_power,
            'power_after_scale': np.mean(np.abs(waveform)**2),
            'scale_factor': scale
        }
        
        return waveform, metadata


class OFDMDemodulator:
    """OFDM Demodulation Engine (FFT + Channel Equalization)"""
    
    def __init__(self, params: OFDMParams):
        self.params = params
    
    def demodulate(self, waveform: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Convert OFDM time-domain waveform → QPSK symbols
        
        Args:
            waveform: Time-domain IQ samples
            
        Returns:
            qpsk_symbols: Recovered QPSK symbols
            metadata: Demodulation info
        """
        # 1. Split waveform into OFDM symbols
        sym_len = self.params.symbol_length
        num_ofdm_symbols = len(waveform) // sym_len
        
        all_qpsk_symbols = []
        channel_estimates = []
        
        for i in range(num_ofdm_symbols):
            # Extract OFDM symbol
            ofdm_symbol = waveform[i * sym_len : (i+1) * sym_len]
            
            # Remove Cyclic Prefix
            ofdm_symbol_no_cp = ofdm_symbol[self.params.cp_len:]
            
            # FFT to frequency domain
            freq_domain = np.fft.fft(ofdm_symbol_no_cp)
            
            # Extract pilots for channel estimation
            pilots_rx = []
            for carrier_idx in self.params.pilot_carriers:
                pilots_rx.append(freq_domain[carrier_idx % self.params.fft_size])
            pilots_rx = np.array(pilots_rx)
            
            # Channel estimation (simple LS)
            h_est = pilots_rx / (self.params.pilot_values + 1e-10)
            # Use average of inner pilots for data region
            h_avg = np.mean(h_est[1:3]) if len(h_est) >= 3 else np.mean(h_est)
            channel_estimates.append(h_avg)
            
            # Extract and equalize data carriers
            data_rx = []
            for carrier_idx in self.params.data_carriers:
                data_rx.append(freq_domain[carrier_idx % self.params.fft_size])
            data_rx = np.array(data_rx)
            
            # Equalize
            if np.abs(h_avg) > 1e-6:
                data_equalized = data_rx / h_avg
            else:
                data_equalized = data_rx
            
            all_qpsk_symbols.extend(data_equalized)
        
        metadata = {
            'num_ofdm_symbols': num_ofdm_symbols,
            'num_qpsk_symbols': len(all_qpsk_symbols),
            'avg_channel_magnitude': np.mean(np.abs(channel_estimates))
        }
        
        return np.array(all_qpsk_symbols, dtype=np.complex64), metadata


class PacketEncoder:
    """
    Handles packetization: Matches GNU Radio 'Stream to Tagged Stream' + 'Stream CRC32'
    
    GNU Radio Logic (from your diagram):
    1. Takes 96 bytes of payload.
    2. Appends 4 bytes of CRC32.
    Total Packet = 100 bytes.
    
    Note: GNU Radio's 'Stream to Tagged Stream' usually DOES NOT add a length header 
    to the data stream itself; it just adds a tag for the next block. 
    But 'Stream CRC32' appends the CRC at the end.
    """
    
    PAYLOAD_LEN = 96
    CRC_LEN = 4
    PACKET_LEN = PAYLOAD_LEN + CRC_LEN
    
    @staticmethod
    def encode(payload: bytes) -> bytes:
        """
        Encodes bytes into fixed-size packets with CRC32.
        Input: Any length bytes
        Output: Multiple 100-byte packets (96 data + 4 CRC)
        """
        packets = []
        
        # Pad payload to multiple of 96
        if len(payload) % PacketEncoder.PAYLOAD_LEN != 0:
            pad_len = PacketEncoder.PAYLOAD_LEN - (len(payload) % PacketEncoder.PAYLOAD_LEN)
            payload += b'\0' * pad_len
            
        # Chunk into 96-byte blocks
        for i in range(0, len(payload), PacketEncoder.PAYLOAD_LEN):
            chunk = payload[i : i + PacketEncoder.PAYLOAD_LEN]
            
            # Calculate CRC32 (Standard IEEE 802.3 polynomial)
            crc = binascii.crc32(chunk) & 0xFFFFFFFF
            crc_bytes = struct.pack('>I', crc)  # Big-endian unsigned int
            
            # Append CRC
            packets.append(chunk + crc_bytes)
            
        return b''.join(packets)
    
    @staticmethod
    def decode(stream: bytes) -> Tuple[Optional[bytes], dict]:
        """
        Decodes a stream of 100-byte packets.
        Validates CRC32 for each.
        """
        valid_data = []
        errors = 0
        total_packets = 0
        
        # Process 100 bytes at a time
        for i in range(0, len(stream), PacketEncoder.PACKET_LEN):
            chunk = stream[i : i + PacketEncoder.PACKET_LEN]
            
            if len(chunk) < PacketEncoder.PACKET_LEN:
                break  # Incomplete packet at end
                
            total_packets += 1
            
            payload = chunk[:PacketEncoder.PAYLOAD_LEN]
            received_crc = struct.unpack('>I', chunk[PacketEncoder.PAYLOAD_LEN:])[0]
            
            # Validate
            calculated_crc = binascii.crc32(payload) & 0xFFFFFFFF
            
            if received_crc == calculated_crc:
                valid_data.append(payload)
            else:
                errors += 1
                # Optional: Keep bad data? Usually no.
                # valid_data.append(b'?' * PacketEncoder.PAYLOAD_LEN) 
        
        # Join valid parts
        decoded_bytes = b''.join(valid_data)
        
        # Attempt to strip null padding (only from the very end)
        # Be careful not to strip valid null bytes inside binary data
        decoded_bytes = decoded_bytes.rstrip(b'\0')
        
        metadata = {
            'status': 'ok' if errors == 0 else 'crc_errors',
            'total_packets': total_packets,
            'crc_errors': errors,
            'packet_error_rate': errors / total_packets if total_packets > 0 else 0
        }
        
        return decoded_bytes, metadata


class OFDMTransceiver:
    """
    Complete OFDM Transmission/Reception Pipeline
    """
    
    def __init__(self, params: OFDMParams = None):
        self.params = params if params else OFDMParams()
        self.qpsk = QPSKModulator()
        self.ofdm_mod = OFDMModulator(self.params)
        self.ofdm_demod = OFDMDemodulator(self.params)
        self.packet = PacketEncoder()
    
    def transmit(self, data: bytes) -> Tuple[np.ndarray, dict]:
        """
        Complete TX Chain: Bytes → QPSK → OFDM → Waveform
        
        Args:
            data: Raw bytes to transmit
            
        Returns:
            waveform: Complex IQ samples ready for SDR
            metadata: Transmission info
        """
        # 1. Packetize
        packet = self.packet.encode(data)
        
        # 2. Bytes to bits
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
        
        # 3. QPSK Modulation
        qpsk_symbols = self.qpsk.modulate(bits)
        
        # 4. OFDM Modulation (IFFT + CP)
        waveform, ofdm_meta = self.ofdm_mod.modulate(qpsk_symbols)
        
        metadata = {
            **ofdm_meta,
            'input_bytes': len(data),
            'packet_bytes': len(packet),
            'num_bits': len(bits),
        }
        
        return waveform, metadata
    
    def receive(self, waveform: np.ndarray) -> Tuple[Optional[bytes], dict]:
        """
        Complete RX Chain: Waveform → OFDM Demod → QPSK → Bytes
        
        Args:
            waveform: Complex IQ samples from SDR (or after AI denoising)
            
        Returns:
            data: Recovered bytes or None if failed
            metadata: Reception info
        """
        # 1. OFDM Demodulation (FFT + Equalization)
        qpsk_symbols, ofdm_meta = self.ofdm_demod.demodulate(waveform)
        
        # 2. QPSK Demodulation
        bits = self.qpsk.demodulate(qpsk_symbols)
        
        # 3. Bits to bytes
        if len(bits) % 8 != 0:
            bits = bits[:-(len(bits) % 8)]
        packet = np.packbits(bits).tobytes()
        
        # 4. De-packetize
        data, packet_meta = self.packet.decode(packet)
        
        metadata = {
            **ofdm_meta,
            **packet_meta,
            'num_bits_recovered': len(bits)
        }
        
        return data, metadata
    
    def plot_constellation(self, waveform: np.ndarray, title="QPSK Constellation"):
        """Plot constellation diagram after OFDM demodulation"""
        qpsk_symbols, _ = self.ofdm_demod.demodulate(waveform)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(np.real(qpsk_symbols), np.imag(qpsk_symbols), 
                   alpha=0.3, s=20, c='blue')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('In-Phase (I)')
        plt.ylabel('Quadrature (Q)')
        plt.grid(True, alpha=0.3)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.axvline(0, color='k', linewidth=0.5)
        
        # Add ideal QPSK points
        ideal = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        plt.scatter(np.real(ideal), np.imag(ideal), 
                   c='red', s=200, marker='x', linewidths=3, 
                   label='Ideal QPSK')
        plt.legend()
        plt.tight_layout()
        plt.show()


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def calculate_ber(original_bits: np.ndarray, recovered_bits: np.ndarray) -> float:
    """Calculate Bit Error Rate"""
    min_len = min(len(original_bits), len(recovered_bits))
    errors = np.sum(original_bits[:min_len] != recovered_bits[:min_len])
    return errors / min_len if min_len > 0 else 1.0


def add_awgn_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add AWGN noise to achieve target SNR"""
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise.astype(np.complex64)
