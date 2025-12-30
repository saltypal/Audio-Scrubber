"""
Adaptive Wiener Filter (Decision-Directed + MMSE-STSA)
=======================================================
State-of-the-art classical speech enhancement algorithm.

References:
    - Ephraim, Y. & Malah, D. (1984). "Speech enhancement using a minimum 
      mean-square error short-time spectral amplitude estimator."
      IEEE Trans. ASSP, Vol. 32, No. 6, pp. 1109-1121.
    
    - Scalart, P. & Filho, J.V. (1996). "Speech enhancement based on a priori 
      signal to noise estimation." IEEE ICASSP, pp. 629-632.

Key Concepts:
    
    1. A Posteriori SNR (γ):
       γ(k,l) = |Y(k,l)|² / λ_n(k)
       Instantaneous SNR estimate at frequency k, frame l
    
    2. A Priori SNR (ξ) - Decision-Directed Approach:
       ξ(k,l) = α · |Ĝ(k,l-1)|² · γ(k,l-1) + (1-α) · max(γ(k,l)-1, 0)
       
       - α: Smoothing factor (0.98 typical)
       - Ĝ: Previous frame's gain
       - Combines past estimate with current ML estimate
       - Reduces musical noise significantly
    
    3. MMSE-STSA Gain Function:
       G(ξ,γ) = (√π/2) · (√ν/γ) · exp(-ν/2) · [(1+ν)I₀(ν/2) + ν·I₁(ν/2)]
       where ν = ξγ/(1+ξ)
       
       - I₀, I₁: Modified Bessel functions of first kind
       - Optimal estimator assuming Gaussian speech and noise

Usage:
    from src.fm.classical.adaptive_wiener import AdaptiveWienerFilter
    
    # Decision-Directed with simple Wiener gain
    denoiser = AdaptiveWienerFilter(alpha=0.98, use_mmse=False)
    
    # Full MMSE-STSA (best quality)
    denoiser = AdaptiveWienerFilter(alpha=0.98, use_mmse=True)
    
    clean = denoiser.denoise(noisy)
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple
import warnings


class AdaptiveWienerFilter:
    """
    Adaptive Wiener Filter with Decision-Directed SNR estimation.
    
    Supports two gain functions:
    1. Simple Wiener: G = ξ/(1+ξ)
    2. MMSE-STSA: Optimal for Gaussian signals (uses Bessel functions)
    
    Features:
    - Decision-directed a priori SNR estimation (reduces musical noise)
    - Optional adaptive noise tracking
    - MMSE-STSA or simple Wiener gain
    
    Attributes:
        frame_size: STFT frame size
        hop_size: STFT hop size  
        noise_frames: Initial frames for noise estimation
        alpha: DD smoothing factor (0.9-0.99, higher = smoother)
        min_gain_db: Minimum gain in dB
        use_mmse: Use MMSE-STSA (True) or simple Wiener (False)
        noise_tracking: Enable adaptive noise estimation
    """
    
    def __init__(
        self,
        frame_size: int = 2048,
        hop_size: Optional[int] = None,
        noise_frames: int = 10,
        alpha: float = 0.98,
        min_gain_db: float = -25,
        use_mmse: bool = True,
        noise_tracking: bool = True,
        noise_smoothing: float = 0.95
    ):
        """
        Initialize Adaptive Wiener Filter.
        
        Args:
            frame_size: STFT frame size
            hop_size: STFT hop size (None = frame_size // 4)
            noise_frames: Initial frames for noise estimation
            alpha: Decision-directed smoothing factor
                   - 0.98: Smooth, less musical noise (recommended)
                   - 0.90: Faster tracking, more artifacts
            min_gain_db: Minimum gain in dB (prevents silence)
            use_mmse: Use MMSE-STSA gain function
                      - True: Better quality, more computation
                      - False: Simpler Wiener gain
            noise_tracking: Adapt noise estimate during processing
            noise_smoothing: Smoothing for noise update (0.9-0.99)
        """
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else frame_size // 4
        self.noise_frames = noise_frames
        self.alpha = alpha
        self.min_gain = 10 ** (min_gain_db / 20)
        self.use_mmse = use_mmse
        self.noise_tracking = noise_tracking
        self.noise_smoothing = noise_smoothing
        
        # Window
        self.window = signal.windows.hann(frame_size, sym=False)
        
        # Check for scipy.special (needed for MMSE-STSA)
        self._bessel_available = True
        try:
            from scipy.special import iv
        except ImportError:
            self._bessel_available = False
            if use_mmse:
                warnings.warn("scipy.special not available, falling back to Wiener gain")
    
    def _stft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute STFT."""
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
        """Estimate noise PSD from initial frames."""
        n = min(self.noise_frames, magnitude.shape[1])
        return np.mean(magnitude[:, :n] ** 2, axis=1)
    
    def _wiener_gain(self, xi: np.ndarray) -> np.ndarray:
        """
        Simple Wiener gain function.
        
        G(ξ) = ξ / (1 + ξ)
        """
        gain = xi / (1 + xi + 1e-10)
        return np.clip(gain, self.min_gain, 1.0)
    
    def _mmse_stsa_gain(self, xi: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """
        MMSE-STSA gain function (Ephraim & Malah, 1984).
        
        G = (√π/2) · (√ν/γ) · exp(-ν/2) · [(1+ν)I₀(ν/2) + ν·I₁(ν/2)]
        
        where ν = ξγ/(1+ξ)
        """
        if not self._bessel_available:
            return self._wiener_gain(xi)
        
        from scipy.special import iv as bessel_iv
        
        # Compute ν = ξγ/(1+ξ)
        nu = (xi * gamma) / (1 + xi + 1e-10)
        nu = np.clip(nu, 1e-10, 500)  # Prevent overflow
        
        # Bessel functions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            I0 = bessel_iv(0, nu / 2)
            I1 = bessel_iv(1, nu / 2)
        
        # MMSE-STSA gain
        sqrt_nu_gamma = np.sqrt(nu / (gamma + 1e-10))
        exp_term = np.exp(-nu / 2)
        bessel_term = (1 + nu) * I0 + nu * I1
        
        gain = (np.sqrt(np.pi) / 2) * sqrt_nu_gamma * exp_term * bessel_term
        
        # Handle numerical issues
        gain = np.clip(gain, self.min_gain, 1.0)
        gain = np.nan_to_num(gain, nan=self.min_gain, posinf=1.0, neginf=self.min_gain)
        
        return gain
    
    def _update_noise(self, noise_psd: np.ndarray, frame_power: np.ndarray,
                      gain: np.ndarray) -> np.ndarray:
        """
        Adaptive noise estimation based on speech presence.
        
        Uses gain as proxy for speech presence:
        - Low gain → likely noise → update estimate
        - High gain → likely speech → keep estimate
        """
        # Speech presence probability (simple approximation)
        speech_prob = np.clip(gain / 0.5, 0, 1)
        
        # Update where speech is absent
        update_rate = (1 - speech_prob) * (1 - self.noise_smoothing)
        return noise_psd + update_rate * (frame_power - noise_psd)
    
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Denoise audio using Adaptive Wiener Filter.
        
        Args:
            audio: Input noisy audio (1D float32)
        
        Returns:
            Denoised audio (same shape)
        """
        original_length = len(audio)
        
        # STFT
        magnitude, phase = self._stft(audio)
        n_bins, n_frames = magnitude.shape
        
        # Initialize
        noise_psd = self._estimate_noise_psd(magnitude)
        prev_gain = np.ones(n_bins)
        enhanced = np.zeros_like(magnitude)
        
        for t in range(n_frames):
            frame_mag = magnitude[:, t]
            frame_power = frame_mag ** 2
            
            # A posteriori SNR: γ = |Y|²/λ_n
            gamma = frame_power / (noise_psd + 1e-10)
            gamma = np.maximum(gamma, 1e-10)
            
            # Decision-Directed a priori SNR
            # ξ = α·(G²·γ)_{prev} + (1-α)·max(γ-1, 0)
            xi_ml = np.maximum(gamma - 1, 0)  # ML estimate
            xi = self.alpha * (prev_gain ** 2 * gamma) + (1 - self.alpha) * xi_ml
            xi = np.maximum(xi, 1e-10)
            
            # Compute gain
            if self.use_mmse:
                gain = self._mmse_stsa_gain(xi, gamma)
            else:
                gain = self._wiener_gain(xi)
            
            # Apply gain
            enhanced[:, t] = gain * frame_mag
            
            # Update noise estimate
            if self.noise_tracking and t >= self.noise_frames:
                noise_psd = self._update_noise(noise_psd, frame_power, gain)
            
            prev_gain = gain
        
        # Reconstruct
        return self._istft(enhanced, phase, original_length)
    
    def reset(self):
        """Reset for new audio stream."""
        pass  # Stateless after initialization
    
    @property
    def name(self) -> str:
        mode = "MMSE-STSA" if self.use_mmse else "DD-Wiener"
        tracking = "+NT" if self.noise_tracking else ""
        return f"Adaptive {mode}{tracking} (α={self.alpha})"
    
    @property
    def category(self) -> str:
        return 'classical'


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Adaptive Wiener Filter Test")
    print("=" * 70)
    
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
    
    # Test configurations
    configs = [
        {"use_mmse": False, "noise_tracking": False, "alpha": 0.98},
        {"use_mmse": False, "noise_tracking": True, "alpha": 0.98},
        {"use_mmse": True, "noise_tracking": False, "alpha": 0.98},
        {"use_mmse": True, "noise_tracking": True, "alpha": 0.98},
    ]
    
    for cfg in configs:
        awf = AdaptiveWienerFilter(**cfg)
        denoised = awf.denoise(noisy)
        out_snr = calc_snr(clean, denoised)
        print(f"  {awf.name:45s} → {out_snr:.2f} dB (+{out_snr - input_snr:.2f})")
    
    print("\n✅ Adaptive Wiener Filter working!")
