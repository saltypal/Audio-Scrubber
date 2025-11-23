"""
Complete OFDM Engine - Matches GNU Radio Training Data Format
Handles text, files, and binary data with full OFDM modulation
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional


class OFDMConfig:
    """OFDM parameters matching dataset_gnu.py exactly."""
    
    FFT_SIZE = 64
    CP_LEN = 16
    
    # GNU Radio uses only 8 data carriers!
    DATA_CARRIERS = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    PILOT_CARRIERS = np.array([-21, -7, 7, 21])
    PILOT_VALUES = np.array([1, 1, 1, -1], dtype=np.complex64)
    
    # QPSK constellation
    QPSK_MAP = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j,
    }
    
    PILOT_SYMBOL = 1 + 1j  # BPSK pilot
    TARGET_POWER = 35.0     # Match training data power


class QPSKMapper:
    """QPSK modulation/demodulation."""
    
    @staticmethod
    def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
        """Convert bits to QPSK symbols."""
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
        
        symbols = []
        for i in range(0, len(bits), 2):
            key = (int(bits[i]), int(bits[i+1]))
            symbols.append(OFDMConfig.QPSK_MAP[key])
        
        return np.array(symbols, dtype=np.complex64)
    
    @staticmethod
    def qpsk_to_bits(symbols: np.ndarray) -> np.ndarray:
        """Hard decision QPSK demodulation."""
        bits = []
        for sym in symbols:
            bit_i = 0 if np.real(sym) > 0 else 1
            bit_q = 0 if np.imag(sym) > 0 else 1
            bits.extend([bit_i, bit_q])
        return np.array(bits, dtype=np.uint8)


class OFDMModulator:
    """
    Full OFDM modulator - converts QPSK symbols to OFDM waveform.
    This creates the exact format your model was trained on.
    """
    
    def __init__(self):
        self.config = OFDMConfig()
    
    def modulate_symbols(self, qpsk_symbols: np.ndarray) -> np.ndarray:
        """
        Convert QPSK symbols to OFDM time-domain waveform.
        
        Args:
            qpsk_symbols: Array of QPSK symbols
            
        Returns:
            OFDM waveform (complex time-domain samples)
        """
        symbols_per_ofdm = len(self.config.DATA_CARRIERS)
        num_ofdm_symbols = int(np.ceil(len(qpsk_symbols) / symbols_per_ofdm))
        
        # Pad to fill last OFDM symbol
        padded = np.pad(
            qpsk_symbols,
            (0, num_ofdm_symbols * symbols_per_ofdm - len(qpsk_symbols)),
            constant_values=0
        )
        
        waveform = []
        
        for i in range(num_ofdm_symbols):
            # Extract chunk for this OFDM symbol
            start = i * symbols_per_ofdm
            end = start + symbols_per_ofdm
            data_chunk = padded[start:end]
            
            # Create one OFDM symbol
            ofdm_symbol = self._create_ofdm_symbol(data_chunk)
            waveform.extend(ofdm_symbol)
        
        waveform = np.array(waveform, dtype=np.complex64)
        
        # Scale to match training data power
        return self._normalize_power(waveform)
    
    def _create_ofdm_symbol(self, data_symbols: np.ndarray) -> np.ndarray:
        """Create one OFDM symbol (frequency → time domain)."""
        # Initialize frequency domain (all zeros)
        freq_domain = np.zeros(self.config.FFT_SIZE, dtype=np.complex64)
        
        # Insert pilots
        for i, idx in enumerate(self.config.PILOT_CARRIERS):
            carrier = idx if idx >= 0 else idx + self.config.FFT_SIZE
            freq_domain[carrier] = self.config.PILOT_VALUES[i]
        
        # Insert data symbols
        for i, carrier_offset in enumerate(self.config.DATA_CARRIERS):
            if i < len(data_symbols):
                carrier = carrier_offset if carrier_offset >= 0 else carrier_offset + self.config.FFT_SIZE
                freq_domain[carrier] = data_symbols[i]
        
        # IFFT: Frequency → Time domain
        # Note: We constructed freq_domain in Standard Order (DC at 0), so we use ifft directly.
        # No ifftshift needed because we manually mapped negative freqs to N-k.
        time_domain = np.fft.ifft(freq_domain)
        
        # Add Cyclic Prefix (copy last CP_LEN samples to beginning)
        cp = time_domain[-self.config.CP_LEN:]
        ofdm_symbol_with_cp = np.concatenate([cp, time_domain])
        
        return ofdm_symbol_with_cp
    
    def _normalize_power(self, signal: np.ndarray) -> np.ndarray:
        """Scale signal to match training data power."""
        current_power = np.mean(np.abs(signal) ** 2)
        if current_power > 0:
            scale = np.sqrt(self.config.TARGET_POWER / current_power)
            return signal * scale
        return signal


class OFDMDemodulator:
    """
    Full OFDM demodulator - converts OFDM waveform back to QPSK symbols.
    """
    
    def __init__(self):
        self.config = OFDMConfig()
    
    def demodulate_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """
        Convert OFDM time-domain waveform to QPSK symbols.
        
        Args:
            waveform: OFDM time-domain samples
            
        Returns:
            QPSK symbols
        """
        symbol_len = self.config.FFT_SIZE + self.config.CP_LEN
        num_symbols = len(waveform) // symbol_len
        
        qpsk_symbols = []
        
        for i in range(num_symbols):
            start = i * symbol_len
            ofdm_symbol = waveform[start:start + symbol_len]
            
            # Extract data from this OFDM symbol
            data = self._extract_data(ofdm_symbol)
            qpsk_symbols.extend(data)
        
        return np.array(qpsk_symbols, dtype=np.complex64)
    
    def _extract_data(self, ofdm_symbol: np.ndarray) -> np.ndarray:
        """Extract QPSK symbols from one OFDM symbol."""
        # Remove cyclic prefix
        time_domain = ofdm_symbol[self.config.CP_LEN:]
        
        # FFT: Time → Frequency domain
        # Note: fft returns Standard Order (DC at 0).
        # We access it using Standard Order indices (N-k for negative), so no fftshift needed.
        freq_domain = np.fft.fft(time_domain)
        
        # Extract data carriers
        data_symbols = []
        for carrier_offset in self.config.DATA_CARRIERS:
            carrier = carrier_offset if carrier_offset >= 0 else carrier_offset + self.config.FFT_SIZE
            data_symbols.append(freq_domain[carrier])
        
        return np.array(data_symbols, dtype=np.complex64)


class OFDMTransmitter:
    """
    Complete OFDM transmitter: Data → Bits → QPSK → OFDM Waveform
    """
    
    def __init__(self):
        self.modulator = OFDMModulator()
        self.qpsk_mapper = QPSKMapper()
    
    def transmit_text(self, text: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Text → OFDM waveform."""
        # Text to bytes
        data_bytes = text.encode('utf-8')
        
        # Add length header (4 bytes)
        length = len(data_bytes)
        packet = length.to_bytes(4, 'big') + data_bytes
        
        # Bytes to bits
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
        
        # Bits to QPSK
        qpsk_symbols = self.qpsk_mapper.bits_to_qpsk(bits)
        
        # QPSK to OFDM waveform
        waveform = self.modulator.modulate_symbols(qpsk_symbols)
        
        metadata = {
            'text_length': len(text),
            'payload_bytes': len(data_bytes),
            'packet_bytes': len(packet),
            'bits': len(bits),
            'qpsk_symbols': len(qpsk_symbols),
            'ofdm_samples': len(waveform),
            'signal_power': float(np.mean(np.abs(waveform) ** 2))
        }
        
        return waveform, metadata
    
    def transmit_bytes(self, data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Binary data → OFDM waveform."""
        # Add length header
        length = len(data)
        packet = length.to_bytes(4, 'big') + data
        
        # Bytes to bits
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
        
        # Bits to QPSK
        qpsk_symbols = self.qpsk_mapper.bits_to_qpsk(bits)
        
        # QPSK to OFDM waveform
        waveform = self.modulator.modulate_symbols(qpsk_symbols)
        
        metadata = {
            'payload_bytes': len(data),
            'packet_bytes': len(packet),
            'bits': len(bits),
            'qpsk_symbols': len(qpsk_symbols),
            'ofdm_samples': len(waveform),
            'signal_power': float(np.mean(np.abs(waveform) ** 2))
        }
        
        return waveform, metadata


class OFDMReceiver:
    """
    Complete OFDM receiver: OFDM Waveform → QPSK → Bits → Data
    """
    
    def __init__(self):
        self.demodulator = OFDMDemodulator()
        self.qpsk_mapper = QPSKMapper()
    
    def receive_text(self, waveform: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """OFDM waveform → Text."""
        # OFDM to QPSK
        qpsk_symbols = self.demodulator.demodulate_waveform(waveform)
        
        # QPSK to bits
        bits = self.qpsk_mapper.qpsk_to_bits(qpsk_symbols)
        
        # Bits to bytes
        byte_array = np.packbits(bits)
        packet_bytes = byte_array.tobytes()
        
        # Parse header
        if len(packet_bytes) < 4:
            return "[ERROR: Packet too short]", {'error': True}
        
        payload_length = int.from_bytes(packet_bytes[:4], 'big')
        payload = packet_bytes[4:4 + payload_length]
        
        # Bytes to text
        try:
            text = payload.decode('utf-8', errors='replace')
        except:
            text = "[DECODE ERROR]"
        
        metadata = {
            'ofdm_samples': len(waveform),
            'qpsk_symbols': len(qpsk_symbols),
            'bits_decoded': len(bits),
            'bytes_decoded': len(packet_bytes),
            'payload_length': payload_length,
            'text_length': len(text)
        }
        
        return text, metadata
    
    def receive_bytes(self, waveform: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """OFDM waveform → Binary data."""
        # OFDM to QPSK
        qpsk_symbols = self.demodulator.demodulate_waveform(waveform)
        
        # QPSK to bits
        bits = self.qpsk_mapper.qpsk_to_bits(qpsk_symbols)
        
        # Bits to bytes
        byte_array = np.packbits(bits)
        packet_bytes = byte_array.tobytes()
        
        # Parse header
        if len(packet_bytes) < 4:
            return b'', {'error': True}
        
        payload_length = int.from_bytes(packet_bytes[:4], 'big')
        payload = packet_bytes[4:4 + payload_length]
        
        metadata = {
            'ofdm_samples': len(waveform),
            'qpsk_symbols': len(qpsk_symbols),
            'bits_decoded': len(bits),
            'bytes_decoded': len(packet_bytes),
            'payload_length': payload_length
        }
        
        return payload, metadata


class ChannelSimulator:
    """Add realistic channel impairments."""
    
    @staticmethod
    def awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add Additive White Gaussian Noise."""
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
        )
        
        return signal + noise


class BERCalculator:
    """Calculate Bit Error Rate."""
    
    @staticmethod
    def calculate_ber(original_waveform: np.ndarray, 
                     received_waveform: np.ndarray) -> float:
        """Calculate BER between two OFDM waveforms."""
        demod = OFDMDemodulator()
        qpsk_mapper = QPSKMapper()
        
        # Demodulate both
        orig_symbols = demod.demodulate_waveform(original_waveform)
        recv_symbols = demod.demodulate_waveform(received_waveform)
        
        # Convert to bits
        orig_bits = qpsk_mapper.qpsk_to_bits(orig_symbols)
        recv_bits = qpsk_mapper.qpsk_to_bits(recv_symbols)
        
        # Compare
        min_len = min(len(orig_bits), len(recv_bits))
        errors = np.sum(orig_bits[:min_len] != recv_bits[:min_len])
        
        return errors / min_len if min_len > 0 else 0.0
