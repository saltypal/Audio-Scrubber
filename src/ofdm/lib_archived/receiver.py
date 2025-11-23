
import numpy as np
from .config import OFDMConfig

class ChannelEqualizer:
    """
    Uses Pilot symbols to estimate and correct channel effects (Phase/Amplitude).
    This is CRITICAL for decoding signals that have been modified by AI or channel.
    """
    
    def __init__(self, config: OFDMConfig):
        self.config = config
    
    def estimate_channel(self, received_pilots: np.ndarray) -> complex:
        """
        Estimate the channel response (H) using Least Squares.
        Uses the two inner pilots (-7, 7) which bracket the data carriers (-4..4).
        """
        # Known transmitted pilots
        tx_pilots = self.config.pilot_values
        
        # H = Rx / Tx
        h_estimates = received_pilots / (tx_pilots + 1e-10)
        
        # We have pilots at [-21, -7, 7, 21]
        # Data is at [-4, -3, -2, -1, 1, 2, 3, 4]
        # We only care about the channel in the data region.
        # Let's average the estimates from -7 and 7 (indices 1 and 2)
        # Note: This assumes pilot_carriers are sorted and match indices 1 and 2.
        # A more robust way would be to interpolate, but this works for this specific config.
        if len(h_estimates) >= 3:
             h_inner = (h_estimates[1] + h_estimates[2]) / 2.0
             return h_inner
        
        return np.mean(h_estimates)
    
    def equalize(self, data_symbols: np.ndarray, h_est: complex) -> np.ndarray:
        """
        Correct data symbols using the estimated channel response.
        Sym_Corrected = Sym_Received / H_est
        """
        if np.abs(h_est) < 1e-6:
            return data_symbols # Avoid blowing up noise if H is near zero
            
        return data_symbols / h_est
