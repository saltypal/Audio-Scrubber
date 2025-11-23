
from dataclasses import dataclass, field
import numpy as np

@dataclass
class OFDMConfig:
    """
    Configuration for OFDM System.
    Default values match the GNU Radio dataset for AI training.
    """
    # System Parameters
    fft_size: int = 64
    cp_len: int = 16
    sample_rate: int = 2000000  # 2 MHz
    
    # Subcarrier Mapping (Indices relative to DC)
    # Using field(default_factory=...) for mutable defaults (numpy arrays)
    data_carriers: np.ndarray = field(default_factory=lambda: np.array([-4, -3, -2, -1, 1, 2, 3, 4]))
    pilot_carriers: np.ndarray = field(default_factory=lambda: np.array([-21, -7, 7, 21]))
    pilot_values: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1, -1], dtype=np.complex64))
    
    # Signal Power Normalization
    target_power: float = 35.0

    @property
    def symbol_len(self) -> int:
        return self.fft_size + self.cp_len
        
    @property
    def data_subcarriers_count(self) -> int:
        return len(self.data_carriers)
