"""
Standard Wiener Filter Denoiser
================================
The optimal linear filter for stationary signals in the MMSE sense.

Theory:
    The Wiener filter minimizes mean square error between the estimated
    and true signal. For stationary signals with known power spectra:
    
    H(ω) = S_xx(ω) / (S_xx(ω) + S_nn(ω))
         = SNR(ω) / (1 + SNR(ω))
    
    where:
        - S_xx(ω): Clean signal power spectral density
        - S_nn(ω): Noise power spectral density
        - SNR(ω): Signal-to-noise ratio at frequency ω

    Since clean signal PSD is unknown, we estimate it as:
        Ŝ_xx(ω) = max(|Y(ω)|² - Ŝ_nn(ω), 0)

Implementation:
    1. Compute STFT of noisy signal
    2. Estimate noise PSD from initial frames
    3. For each frame:
       - Estimate clean signal PSD via spectral subtraction
       - Compute Wiener gain: G = Ŝ_xx / (Ŝ_xx + Ŝ_nn)
       - Apply spectral floor for stability
    4. Reconstruct via ISTFT

Usage:
    from src.fm.classical.wiener_filter import WienerFilter
    
    denoiser = WienerFilter(alpha=1.0, beta=0.02)
    clean = denoiser.denoise(noisy)
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple


class WienerFilter:
    """
    Standard Wiener Filter for audio denoising.
    
    Implements the frequency-domain Wiener filter with:
    - Noise PSD estimation from initial frames
    - Spectral subtraction for clean signal PSD estimation
    - Spectral flooring to prevent musical noise
    
    Attributes:
        frame_size: STFT frame size
        hop_size: STFT hop size
        noise_frames: Frames for noise estimation
        alpha: Noise overestimation factor
        beta: Spectral floor (minimum gain)
    """
    
    def __init__(
        self,
        frame_size: int = 2048,
        hop_size: Optional[int] = None,
        noise_frames: int = 10,
        alpha: float = 1.0,
        beta: float = 0.02
    ):
        """
        Initialize Wiener Filter.
        
        Args:
            frame_size: STFT frame/FFT size
            hop_size: STFT hop size (None = frame_size // 4)
            noise_frames: Number of initial frames for noise estimation
            alpha: Noise overestimation factor (≥1.0)
                   - 1.0: Standard Wiener filter
                   - >1.0: More aggressive noise reduction
            beta: Spectral floor (minimum Wiener gain, 0-1)
                  Prevents excessive attenuation and musical noise
        """
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else frame_size // 4
        self.noise_frames = noise_frames
        self.alpha = alpha
        self.beta = beta
        
        # Hann window
        self.window = signal.windows.hann(frame_size, sym=False)
        
        # For adaptive noise tracking (optional)
        self.noise_psd = None
    
    def _stft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute STFT, returning magnitude and phase."""
        pad_len = (self.frame_size - len(audio) % self.hop_size) % self.hop_size
        audio_padded = np.pad(audio, (0, pad_len))
        
        _, _, Zxx = signal.stft(
            audio_padded,
            nperseg=self.frame_size,
            noverlap=self.frame_size - self.hop_size,
            window=self.window,
            return_onesided=True
        )
        return np.abs(Zxx), np.angle(Zxx)
    
    def _istft(self, magnitude: np.ndarray, phase: np.ndarray,
               original_length: int) -> np.ndarray:
        """Inverse STFT."""
        Zxx = magnitude * np.exp(1j * phase)
        _, audio = signal.istft(
            Zxx,
            nperseg=self.frame_size,
            noverlap=self.frame_size - self.hop_size,
            window=self.window
        )
        return audio[:original_length].astype(np.float32)
    
    def _estimate_noise_psd(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Estimate noise PSD from initial frames.
        
        Assumes first frames are noise-only or noise-dominated.
        """
        n = min(self.noise_frames, magnitude.shape[1])
        noise_mag = magnitude[:, :n]
        return np.mean(noise_mag ** 2, axis=1)
    
    def _wiener_gain(self, signal_psd: np.ndarray, 
                     noise_psd: np.ndarray) -> np.ndarray:
        """
        Compute Wiener filter gain.
        
        G = Ŝ_xx / (Ŝ_xx + Ŝ_nn)
        
        where Ŝ_xx = max(|Y|² - α·Ŝ_nn, 0)
        """
        # Estimate clean signal PSD via spectral subtraction
        clean_psd_est = np.maximum(signal_psd - self.alpha * noise_psd, 0)
        
        # Wiener gain
        gain = clean_psd_est / (clean_psd_est + noise_psd + 1e-10)
        
        # Apply spectral floor
        gain = np.maximum(gain, self.beta)
        
        return gain
    
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Denoise audio using Wiener filtering.
        
        Args:
            audio: Input noisy audio (1D float32)
        
        Returns:
            Denoised audio (same shape)
        """
        original_length = len(audio)
        
        # STFT
        magnitude, phase = self._stft(audio)
        
        # Estimate noise PSD
        if self.noise_psd is None:
            noise_psd = self._estimate_noise_psd(magnitude)
        else:
            noise_psd = self.noise_psd
        
        # Reshape for broadcasting (n_bins, 1)
        noise_psd_2d = noise_psd[:, np.newaxis]
        
        # Signal PSD
        signal_psd = magnitude ** 2
        
        # Compute Wiener gain for all frames
        gain = self._wiener_gain(signal_psd, noise_psd_2d)
        
        # Apply gain
        enhanced_magnitude = magnitude * gain
        
        # Reconstruct
        return self._istft(enhanced_magnitude, phase, original_length)
    
    def denoise_with_noise_sample(self, audio: np.ndarray, 
                                  noise_sample: np.ndarray) -> np.ndarray:
        """
        Denoise using a separate noise sample for PSD estimation.
        
        More accurate when you have a clean noise sample.
        
        Args:
            audio: Noisy audio to denoise
            noise_sample: Sample of pure noise
        
        Returns:
            Denoised audio
        """
        noise_mag, _ = self._stft(noise_sample)
        self.noise_psd = np.mean(noise_mag ** 2, axis=1)
        return self.denoise(audio)
    
    def set_noise_psd(self, noise_psd: np.ndarray):
        """Set pre-computed noise PSD."""
        self.noise_psd = noise_psd
    
    def reset(self):
        """Reset noise estimate for new audio stream."""
        self.noise_psd = None
    
    @property
    def name(self) -> str:
        return f"Wiener Filter (α={self.alpha}, β={self.beta})"
    
    @property
    def category(self) -> str:
        return 'classical'


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Standard Wiener Filter Test")
    print("=" * 60)
    
    np.random.seed(42)
    sr = 44100
    t = np.linspace(0, 1, sr)
    
    clean = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    noise = np.random.randn(len(t)) * 0.1
    noisy = (clean + noise).astype(np.float32)
    
    def calc_snr(c, s):
        n = s - c
        return 10 * np.log10(np.mean(c**2) / (np.mean(n**2) + 1e-10))
    
    input_snr = calc_snr(clean, noisy)
    print(f"\nInput SNR: {input_snr:.2f} dB\n")
    
    # Test different configurations
    configs = [
        {"alpha": 1.0, "beta": 0.02},  # Standard
        {"alpha": 1.5, "beta": 0.02},  # Moderate
        {"alpha": 2.0, "beta": 0.01},  # Aggressive
    ]
    
    for cfg in configs:
        wf = WienerFilter(**cfg)
        denoised = wf.denoise(noisy)
        out_snr = calc_snr(clean, denoised)
        print(f"  {wf.name:40s} → {out_snr:.2f} dB (+{out_snr - input_snr:.2f})")
    
    print("\n✅ Wiener Filter working!")
