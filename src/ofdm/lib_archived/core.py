
import numpy as np
from .config import OFDMConfig

class OFDMEngine:
    """
    Core OFDM Signal Processing Engine.
    Handles IFFT/FFT, Cyclic Prefix, and Resource Mapping.
    """
    
    def __init__(self, config: OFDMConfig):
        self.config = config
    
    def generate_symbol(self, data_chunk: np.ndarray) -> np.ndarray:
        """
        Generate a single time-domain OFDM symbol from data symbols.
        """
        # 1. Create empty frequency domain grid
        freq_domain = np.zeros(self.config.fft_size, dtype=np.complex64)
        
        # 2. Map Pilots
        for i, carrier_idx in enumerate(self.config.pilot_carriers):
            # Handle negative indices (e.g., -1 becomes 63)
            idx = carrier_idx if carrier_idx >= 0 else carrier_idx + self.config.fft_size
            freq_domain[idx] = self.config.pilot_values[i]
            
        # 3. Map Data
        for i, carrier_idx in enumerate(self.config.data_carriers):
            if i < len(data_chunk):
                idx = carrier_idx if carrier_idx >= 0 else carrier_idx + self.config.fft_size
                freq_domain[idx] = data_chunk[i]
        
        # Debug check
        if np.sum(np.abs(freq_domain)) == 0:
             print("DEBUG: freq_domain is all zeros!")
                
        # 4. IFFT (Frequency -> Time)
        # numpy.fft.ifft assumes standard order (DC at 0), which we constructed.
        time_domain = np.fft.ifft(freq_domain)
        
        # 5. Add Cyclic Prefix
        cp = time_domain[-self.config.cp_len:]
        symbol_with_cp = np.concatenate([cp, time_domain])
        
        return symbol_with_cp

    def process_received_symbol(self, time_symbol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Process a single received OFDM symbol.
        Returns: (data_symbols, pilot_symbols)
        """
        if len(time_symbol) != self.config.symbol_len:
            raise ValueError(f"Expected symbol length {self.config.symbol_len}, got {len(time_symbol)}")
            
        # 1. Remove Cyclic Prefix
        signal_no_cp = time_symbol[self.config.cp_len:]
        
        # 2. FFT (Time -> Frequency)
        freq_domain = np.fft.fft(signal_no_cp)
        
        # 3. Extract Data
        data_extracted = []
        for carrier_idx in self.config.data_carriers:
            idx = carrier_idx if carrier_idx >= 0 else carrier_idx + self.config.fft_size
            data_extracted.append(freq_domain[idx])
            
        # 4. Extract Pilots (for channel estimation)
        pilots_extracted = []
        for carrier_idx in self.config.pilot_carriers:
            idx = carrier_idx if carrier_idx >= 0 else carrier_idx + self.config.fft_size
            pilots_extracted.append(freq_domain[idx])
            
        return np.array(data_extracted), np.array(pilots_extracted)
