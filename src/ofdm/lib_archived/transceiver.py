
import numpy as np
from typing import Tuple, Dict, Any
from .config import OFDMConfig
from .modulation import QPSK
from .core import OFDMEngine
from .receiver import ChannelEqualizer

class OFDMTransmitter:
    def __init__(self, config: OFDMConfig = None):
        self.config = config if config else OFDMConfig()
        self.modulator = QPSK()
        self.engine = OFDMEngine(self.config)
        
    def transmit(self, payload_bytes: bytes) -> Tuple[np.ndarray, Dict]:
        """
        Full Transmission Chain: Bytes -> Bits -> QPSK -> OFDM Waveform
        """
        # 1. Prepare Packet (Length Header + Data)
        length_header = len(payload_bytes).to_bytes(4, 'big')
        full_packet = length_header + payload_bytes
        
        # 2. Bytes to Bits
        bits = np.unpackbits(np.frombuffer(full_packet, dtype=np.uint8))
        
        # 3. Bits to Symbols
        symbols = self.modulator.modulate(bits)
        
        # 4. Pad to fill OFDM symbols
        capacity = self.config.data_subcarriers_count
        num_ofdm_symbols = int(np.ceil(len(symbols) / capacity))
        total_capacity = num_ofdm_symbols * capacity
        
        if len(symbols) < total_capacity:
            padding = np.zeros(total_capacity - len(symbols), dtype=np.complex64)
            symbols = np.concatenate([symbols, padding])
            
        # 5. Generate Waveform
        waveform = []
        for i in range(num_ofdm_symbols):
            chunk = symbols[i*capacity : (i+1)*capacity]
            symbol = self.engine.generate_symbol(chunk)
            waveform.extend(symbol)
            
        waveform = np.array(waveform, dtype=np.complex64)
        
        # 6. Power Normalization (Match Training Data)
        current_pwr = np.mean(np.abs(waveform)**2)
        # print(f"DEBUG: Pre-scale power: {current_pwr}")
        
        scale = np.sqrt(self.config.target_power / current_pwr) if current_pwr > 0 else 1.0
        waveform *= scale
        
        meta = {
            "payload_len": len(payload_bytes),
            "num_ofdm_symbols": num_ofdm_symbols,
            "raw_power": current_pwr,
            "scale_factor": scale
        }
        return waveform, meta

class OFDMReceiver:
    def __init__(self, config: OFDMConfig = None):
        self.config = config if config else OFDMConfig()
        self.modulator = QPSK()
        self.engine = OFDMEngine(self.config)
        self.equalizer = ChannelEqualizer(self.config)
        
    def receive(self, waveform: np.ndarray) -> Tuple[bytes, Dict]:
        """
        Full Receiver Chain: Waveform -> FFT -> Equalize -> QPSK -> Bits -> Bytes
        """
        # 1. Split into symbols
        sym_len = self.config.symbol_len
        num_symbols = len(waveform) // sym_len
        
        all_data_symbols = []
        
        for i in range(num_symbols):
            # Extract time domain symbol
            time_sym = waveform[i*sym_len : (i+1)*sym_len]
            
            # Core Processing (CP Removal + FFT)
            raw_data, raw_pilots = self.engine.process_received_symbol(time_sym)
            
            # Channel Estimation & Equalization
            h_est = self.equalizer.estimate_channel(raw_pilots)
            corrected_data = self.equalizer.equalize(raw_data, h_est)
            
            all_data_symbols.extend(corrected_data)
            
        all_data_symbols = np.array(all_data_symbols)
        
        # 2. Demodulate
        bits = self.modulator.demodulate(all_data_symbols)
        
        # 3. Bits to Bytes
        bytes_data = np.packbits(bits).tobytes()
        
        # 4. Parse Header
        try:
            if len(bytes_data) < 4:
                return b"", {"error": "Packet too short"}
                
            payload_len = int.from_bytes(bytes_data[:4], 'big')
            
            if payload_len > len(bytes_data) - 4:
                return b"", {"error": "Invalid length header"}
                
            payload = bytes_data[4 : 4+payload_len]
            return payload, {"status": "ok", "len": payload_len}
            
        except Exception as e:
            return b"", {"error": str(e)}
