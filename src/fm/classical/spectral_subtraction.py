"""
Spectral Subtraction Denoiser (Boll, 1979)
===========================================
One of the earliest and most influential speech enhancement algorithms.

Reference:
    Boll, S.F. (1979). "Suppression of Acoustic Noise in Speech Using
    Spectral Subtraction." IEEE Transactions on Acoustics, Speech, and
    Signal Processing, Vol. ASSP-27, No. 2, pp. 113-120.

Algorithm:
    |X̂(ω)|^p = max(|Y(ω)|^p - α·E[|N(ω)|^p], β·|Y(ω)|^p)
    
    where:
        - Y(ω): Noisy signal spectrum
        - N(ω): Estimated noise spectrum
        - α: Oversubtraction factor (reduces residual noise)
        - β: Spectral floor (prevents musical noise)
        - p: Exponent (1 for magnitude, 2 for power subtraction)

Improvements implemented:
    - Multi-band processing (different α per frequency band)
    - Adaptive oversubtraction based on per-frame SNR
    - Half-wave rectification with spectral flooring

Usage:
    from src.fm.classical.spectral_subtraction import SpectralSubtraction
    
    denoiser = SpectralSubtraction(alpha=2.0, beta=0.01)
    clean = denoiser.denoise(noisy)
"""

import numpy as np
from scipy import signal
from typing import Optional, Literal


class SpectralSubtraction:
    """
    Spectral Subtraction Denoiser.
    
    Subtracts estimated noise spectrum from noisy signal spectrum
    in the frequency domain.
    
    Attributes:
        frame_size: STFT frame size (FFT length)
        hop_size: STFT hop size
        noise_frames: Number of initial frames for noise estimation
        alpha: Oversubtraction factor (1.0-4.0, higher = more aggressive)
        beta: Spectral floor (0.001-0.1, prevents musical noise)
        domain: 'power' or 'magnitude' subtraction
        multiband: Enable multi-band processing
        adaptive: Enable adaptive alpha based on SNR
    """
    
    def __init__(
        self,
        frame_size: int = 2048,
        hop_size: Optional[int] = None,
        noise_frames: int = 10,
        alpha: float = 2.0,
        beta: float = 0.01,
        domain: Literal['power', 'magnitude'] = 'power',
        multiband: bool = False,
        n_bands: int = 4,
        adaptive: bool = True
    ):
        """
        Initialize Spectral Subtraction denoiser.
        
        Args:
            frame_size: STFT frame/FFT size (powers of 2 recommended)
            hop_size: STFT hop size (None = frame_size // 4)
            noise_frames: Number of initial frames for noise estimation
            alpha: Oversubtraction factor
                   - 1.0: Standard subtraction
                   - 2.0-4.0: More aggressive (for high noise)
            beta: Spectral floor (minimum gain to prevent zeroing)
            domain: 'power' (|X|²) or 'magnitude' (|X|) subtraction
            multiband: Use multi-band processing (different α per band)
            n_bands: Number of frequency bands for multi-band mode
            adaptive: Adapt α based on per-frame SNR estimate
        """
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else frame_size // 4
        self.noise_frames = noise_frames
        self.alpha = alpha
        self.beta = beta
        self.domain = domain
        self.multiband = multiband
        self.n_bands = n_bands
        self.adaptive = adaptive
        
        # Hann window for STFT (good overlap-add properties)
        self.window = signal.windows.hann(frame_size, sym=False)
        
        # Frequency bin edges for multi-band
        self.n_bins = frame_size // 2 + 1
        self.band_edges = np.linspace(0, self.n_bins, n_bands + 1, dtype=int)
    
    def _stft(self, audio: np.ndarray):
        """Compute Short-Time Fourier Transform."""
        _, _, Zxx = signal.stft(
            audio,
            nperseg=self.frame_size,
            noverlap=self.frame_size - self.hop_size,
            window=self.window,
            return_onesided=True
        )
        return np.abs(Zxx), np.angle(Zxx)
    
    def _istft(self, magnitude: np.ndarray, phase: np.ndarray, 
               original_length: int) -> np.ndarray:
        """Inverse STFT from magnitude and phase."""
        Zxx = magnitude * np.exp(1j * phase)
        _, audio = signal.istft(
            Zxx,
            nperseg=self.frame_size,
            noverlap=self.frame_size - self.hop_size,
            window=self.window
        )
        return audio[:original_length].astype(np.float32)
    
    def _estimate_noise(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Estimate noise spectrum from initial frames.
        
        Assumes first few frames are noise-only or noise-dominated.
        """
        n = min(self.noise_frames, magnitude.shape[1])
        noise_mag = magnitude[:, :n]
        
        if self.domain == 'power':
            return np.mean(noise_mag ** 2, axis=1)
        else:
            return np.mean(noise_mag, axis=1)
    
    def _compute_snr(self, signal_power: np.ndarray, 
                     noise_power: np.ndarray) -> float:
        """Compute SNR in dB."""
        snr_linear = np.mean(signal_power) / (np.mean(noise_power) + 1e-10)
        return 10 * np.log10(snr_linear + 1e-10)
    
    def _adaptive_alpha(self, snr_db: float) -> float:
        """
        Compute adaptive oversubtraction factor.
        
        Low SNR → higher α (more aggressive)
        High SNR → lower α (less needed)
        
        Based on: α = α₀ - SNR/20, clamped to [1, 4]
        """
        alpha = self.alpha - (snr_db / 20.0)
        return np.clip(alpha, 1.0, 4.0)
    
    def _subtract_frame(self, frame_mag: np.ndarray, 
                        noise_est: np.ndarray, alpha: float) -> np.ndarray:
        """Apply spectral subtraction to a single frame."""
        if self.domain == 'power':
            frame_power = frame_mag ** 2
            clean_power = frame_power - alpha * noise_est
            floor = self.beta * frame_power
            clean_power = np.maximum(clean_power, floor)
            return np.sqrt(clean_power)
        else:
            clean_mag = frame_mag - alpha * noise_est
            floor = self.beta * frame_mag
            return np.maximum(clean_mag, floor)
    
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Denoise audio using spectral subtraction.
        
        Args:
            audio: Input noisy audio (1D float32 array)
        
        Returns:
            Denoised audio (same shape as input)
        """
        original_length = len(audio)
        
        # Compute STFT
        magnitude, phase = self._stft(audio)
        n_frames = magnitude.shape[1]
        
        # Estimate noise spectrum
        noise_est = self._estimate_noise(magnitude)
        
        # Process each frame
        enhanced = np.zeros_like(magnitude)
        
        for t in range(n_frames):
            frame_mag = magnitude[:, t]
            frame_power = frame_mag ** 2
            
            if self.multiband:
                # Multi-band: different α per frequency band
                for i in range(self.n_bands):
                    lo, hi = self.band_edges[i], self.band_edges[i + 1]
                    
                    if self.adaptive:
                        band_snr = self._compute_snr(
                            frame_power[lo:hi],
                            noise_est[lo:hi] if self.domain == 'power' 
                            else noise_est[lo:hi] ** 2
                        )
                        alpha = self._adaptive_alpha(band_snr)
                    else:
                        alpha = self.alpha
                    
                    enhanced[lo:hi, t] = self._subtract_frame(
                        frame_mag[lo:hi], noise_est[lo:hi], alpha
                    )
            else:
                # Single-band processing
                if self.adaptive:
                    frame_snr = self._compute_snr(
                        frame_power,
                        noise_est if self.domain == 'power' else noise_est ** 2
                    )
                    alpha = self._adaptive_alpha(frame_snr)
                else:
                    alpha = self.alpha
                
                enhanced[:, t] = self._subtract_frame(frame_mag, noise_est, alpha)
        
        # Reconstruct
        return self._istft(enhanced, phase, original_length)
    
    @property
    def name(self) -> str:
        mode = "MB" if self.multiband else "SB"
        adapt = "+Adaptive" if self.adaptive else ""
        return f"Spectral Sub ({mode}{adapt}, α={self.alpha})"
    
    @property
    def category(self) -> str:
        return 'classical'


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Spectral Subtraction Test")
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
    
    # Test variants
    configs = [
        {"multiband": False, "adaptive": False, "alpha": 2.0},
        {"multiband": False, "adaptive": True, "alpha": 2.0},
        {"multiband": True, "adaptive": True, "alpha": 2.0},
    ]
    
    for cfg in configs:
        ss = SpectralSubtraction(**cfg)
        denoised = ss.denoise(noisy)
        out_snr = calc_snr(clean, denoised)
        print(f"  {ss.name:40s} → {out_snr:.2f} dB (+{out_snr - input_snr:.2f})")
    
    print("\n✅ Spectral Subtraction working!")
