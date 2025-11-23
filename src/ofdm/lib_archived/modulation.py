
import numpy as np
from abc import ABC, abstractmethod

class Modulator(ABC):
    """Abstract Base Class for Modulation Schemes."""
    
    @abstractmethod
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Convert bits to complex symbols."""
        pass
        
    @abstractmethod
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Convert complex symbols to bits."""
        pass
    
    @property
    @abstractmethod
    def bits_per_symbol(self) -> int:
        pass

class QPSK(Modulator):
    """
    QPSK Modulation and Demodulation.
    Maps bits to complex symbols and vice versa.
    """
    
    # Mapping table: (bit1, bit0) -> complex symbol
    # Matches standard QPSK: 00->1+j, 01->-1+j, 11->-1-j, 10->1-j
    MAP = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j,
    }
    
    @property
    def bits_per_symbol(self) -> int:
        return 2
    
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Convert bits (0/1) to QPSK symbols."""
        # Ensure even number of bits
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
            
        symbols = []
        for i in range(0, len(bits), 2):
            key = (int(bits[i]), int(bits[i+1]))
            symbols.append(self.MAP[key])
            
        return np.array(symbols, dtype=np.complex64)
    
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        Hard decision demodulation.
        Maps complex symbols back to bits based on quadrant.
        Based on MAP:
        (0,0) -> 1+1j  (I>0, Q>0)
        (0,1) -> -1+1j (I<0, Q>0)
        (1,1) -> -1-1j (I<0, Q<0)
        (1,0) -> 1-1j  (I>0, Q<0)
        
        Logic:
        First Bit (b0): 0 if Q>0, 1 if Q<0
        Second Bit (b1): 0 if I>0, 1 if I<0
        """
        bits = []
        for sym in symbols:
            # First bit depends on Imaginary part
            b0 = 0 if sym.imag > 0 else 1
            # Second bit depends on Real part
            b1 = 0 if sym.real > 0 else 1
            bits.extend([b0, b1])
            
        return np.array(bits, dtype=np.uint8)
