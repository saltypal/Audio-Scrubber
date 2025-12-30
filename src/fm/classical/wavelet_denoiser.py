"""
Wavelet Thresholding Denoiser for FM Audio
============================================
Classical signal processing approach using discrete wavelet transform (DWT).
Supports soft/hard thresholding with universal or SURE threshold estimation.

Usage:
    from src.fm.classical.wavelet_denoiser import WaveletDenoiser
    
    denoiser = WaveletDenoiser(wavelet='db4', level=5, threshold_mode='soft')
    clean_audio = denoiser.denoise(noisy_audio)
"""

import numpy as np
import pywt
from typing import Literal, Optional


class WaveletDenoiser:
    """
    Wavelet-based audio denoiser using thresholding.
    
    Attributes:
        wavelet: Wavelet family (e.g., 'db4', 'sym8', 'coif3')
        level: Decomposition level (None = max level)
        threshold_mode: 'soft' or 'hard' thresholding
        threshold_rule: 'universal', 'sure', 'heursure', 'minimax'
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        level: Optional[int] = None,
        threshold_mode: Literal['soft', 'hard'] = 'soft',
        threshold_rule: Literal['universal', 'sure', 'heursure', 'minimax'] = 'universal',
        noise_sigma: Optional[float] = None
    ):
        """
        Initialize wavelet denoiser.
        
        Args:
            wavelet: Wavelet name from pywt.wavelist()
            level: Decomposition level (None for automatic)
            threshold_mode: 'soft' (shrinkage) or 'hard' (keep/zero)
            threshold_rule: Threshold estimation method
            noise_sigma: Known noise std; if None, estimate from signal
        """
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode
        self.threshold_rule = threshold_rule
        self.noise_sigma = noise_sigma
        
        # Validate wavelet
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Unknown wavelet '{wavelet}'. Choose from: {pywt.wavelist()[:20]}...")
    
    def _estimate_noise_sigma(self, detail_coeffs: np.ndarray) -> float:
        """
        Estimate noise standard deviation using MAD (Median Absolute Deviation).
        Uses the finest detail coefficients (highest frequency).
        """
        # MAD estimator: sigma = MAD / 0.6745
        mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
        return mad / 0.6745
    
    def _compute_threshold(self, coeffs: np.ndarray, sigma: float, n: int) -> float:
        """
        Compute threshold based on selected rule.
        
        Args:
            coeffs: Wavelet coefficients
            sigma: Noise standard deviation
            n: Signal length
        
        Returns:
            Threshold value
        """
        if self.threshold_rule == 'universal':
            # Universal (VisuShrink): sigma * sqrt(2 * log(n))
            return sigma * np.sqrt(2 * np.log(n))
        
        elif self.threshold_rule == 'sure':
            # SURE (Stein's Unbiased Risk Estimate)
            return self._sure_threshold(coeffs, sigma)
        
        elif self.threshold_rule == 'heursure':
            # Heuristic SURE: use universal if sparsity is high
            eta = (np.sum(coeffs ** 2) - n) / n
            crit = (np.log2(n) ** 1.5) / np.sqrt(n)
            if eta < crit:
                return sigma * np.sqrt(2 * np.log(n))
            else:
                return min(self._sure_threshold(coeffs, sigma),
                          sigma * np.sqrt(2 * np.log(n)))
        
        elif self.threshold_rule == 'minimax':
            # Minimax threshold (lookup table approximation)
            if n > 32:
                return sigma * (0.3936 + 0.1829 * np.log2(n))
            else:
                return 0.0
        
        else:
            raise ValueError(f"Unknown threshold rule: {self.threshold_rule}")
    
    def _sure_threshold(self, coeffs: np.ndarray, sigma: float) -> float:
        """
        Compute SURE (Stein's Unbiased Risk Estimate) threshold.
        """
        n = len(coeffs)
        coeffs_sorted = np.sort(np.abs(coeffs))
        
        # Compute SURE risk for each possible threshold
        risks = np.zeros(n)
        for i, t in enumerate(coeffs_sorted):
            # Number of coefficients below threshold
            k = np.sum(np.abs(coeffs) <= t)
            # SURE formula
            risks[i] = n - 2 * k + np.sum(np.minimum(coeffs ** 2, t ** 2))
        
        # Return threshold with minimum risk
        best_idx = np.argmin(risks)
        return coeffs_sorted[best_idx]
    
    def _apply_threshold(self, coeffs: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply soft or hard thresholding to coefficients.
        """
        if self.threshold_mode == 'soft':
            # Soft thresholding: shrink towards zero
            return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
        else:
            # Hard thresholding: keep or zero
            return coeffs * (np.abs(coeffs) > threshold)
    
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Denoise audio signal using wavelet thresholding.
        
        Args:
            audio: Input noisy audio (1D numpy array, float32)
        
        Returns:
            Denoised audio (same shape as input)
        """
        n = len(audio)
        
        # Determine decomposition level
        if self.level is None:
            max_level = pywt.dwt_max_level(n, pywt.Wavelet(self.wavelet).dec_len)
            level = min(max_level, 6)  # Cap at 6 for performance
        else:
            level = self.level
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(audio, self.wavelet, level=level)
        
        # Estimate noise sigma from finest detail coefficients
        if self.noise_sigma is None:
            sigma = self._estimate_noise_sigma(coeffs[-1])
        else:
            sigma = self.noise_sigma
        
        # Apply thresholding to detail coefficients (not approximation)
        denoised_coeffs = [coeffs[0]]  # Keep approximation unchanged
        
        for i, detail in enumerate(coeffs[1:]):
            threshold = self._compute_threshold(detail, sigma, n)
            denoised_detail = self._apply_threshold(detail, threshold)
            denoised_coeffs.append(denoised_detail)
        
        # Reconstruct signal
        denoised = pywt.waverec(denoised_coeffs, self.wavelet)
        
        # Ensure same length as input (waverec can add samples)
        if len(denoised) > n:
            denoised = denoised[:n]
        elif len(denoised) < n:
            denoised = np.pad(denoised, (0, n - len(denoised)))
        
        return denoised.astype(np.float32)
    
    def denoise_chunked(self, audio: np.ndarray, chunk_size: int = 8192,
                        overlap: int = 1024) -> np.ndarray:
        """
        Denoise audio in overlapping chunks (for real-time processing).
        
        Args:
            audio: Input audio
            chunk_size: Size of each chunk
            overlap: Overlap between consecutive chunks
        
        Returns:
            Denoised audio
        """
        n = len(audio)
        hop = chunk_size - overlap
        output = np.zeros(n, dtype=np.float32)
        weights = np.zeros(n, dtype=np.float32)
        
        # Create fade window for smooth transitions
        fade_in = np.linspace(0, 1, overlap)
        fade_out = np.linspace(1, 0, overlap)
        window = np.ones(chunk_size)
        window[:overlap] = fade_in
        window[-overlap:] = fade_out
        
        for start in range(0, n - chunk_size + 1, hop):
            chunk = audio[start:start + chunk_size]
            denoised_chunk = self.denoise(chunk)
            
            # Apply window and accumulate
            output[start:start + chunk_size] += denoised_chunk * window
            weights[start:start + chunk_size] += window
        
        # Handle last chunk if needed
        if (n - chunk_size) % hop != 0:
            start = n - chunk_size
            chunk = audio[start:]
            if len(chunk) == chunk_size:
                denoised_chunk = self.denoise(chunk)
                output[start:] += denoised_chunk * window
                weights[start:] += window
        
        # Normalize by weights
        weights = np.maximum(weights, 1e-8)
        output /= weights
        
        return output
    
    @property
    def name(self) -> str:
        """Return descriptive name for UI/logging."""
        return f"Wavelet ({self.wavelet}, {self.threshold_mode})"
    
    @property
    def category(self) -> str:
        """Denoiser category for UI grouping."""
        return 'classical'
    
    @staticmethod
    def list_wavelets() -> list:
        """List available wavelet families."""
        return pywt.wavelist()


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("Wavelet Denoiser Test")
    print("=" * 60)
    
    # Generate test signal
    np.random.seed(42)
    t = np.linspace(0, 1, 44100)
    clean = np.sin(2 * np.pi * 440 * t) * 0.5
    noise = np.random.randn(len(t)) * 0.1
    noisy = clean + noise
    
    # Denoise
    denoiser = WaveletDenoiser(wavelet='db4', threshold_mode='soft')
    denoised = denoiser.denoise(noisy.astype(np.float32))
    
    # Calculate SNR
    def snr(clean, noisy):
        noise = noisy - clean
        return 10 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2))
    
    print(f"Input SNR:  {snr(clean, noisy):.2f} dB")
    print(f"Output SNR: {snr(clean, denoised):.2f} dB")
    print(f"SNR Gain:   {snr(clean, denoised) - snr(clean, noisy):.2f} dB")
    print("\n✅ Wavelet denoiser working!")
