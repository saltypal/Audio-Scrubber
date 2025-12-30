"""
Wiener Filter Denoiser for FM Audio
====================================
Classical signal processing approach using frequency-domain Wiener filtering.
Estimates noise spectrum and applies optimal linear filter.

Usage:
    from src.fm.classical.wiener_denoiser import WienerDenoiser
    
    denoiser = WienerDenoiser(frame_size=2048, noise_frames=10)
    clean_audio = denoiser.denoise(noisy_audio)
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple


class WienerDenoiser:
    """
    Wiener filter audio denoiser.
    
    The Wiener filter is the optimal linear filter for stationary signals
    in the mean-square error sense. It estimates the clean signal as:
    
        H(f) = S_xx(f) / (S_xx(f) + S_nn(f))
    
    where S_xx is the clean signal PSD and S_nn is the noise PSD.
    
    Attributes:
        frame_size: STFT frame size
        hop_size: STFT hop size
        noise_frames: Number of initial frames to estimate noise
        alpha: Oversubtraction factor (1.0 = standard, >1.0 = aggressive)
        beta: Spectral floor (prevents musical noise)
    """
    
    def __init__(
        self,
        frame_size: int = 2048,
        hop_size: Optional[int] = None,
        noise_frames: int = 10,
        alpha: float = 1.0,
        beta: float = 0.02,
        noise_psd: Optional[np.ndarray] = None
    ):
        """
        Initialize Wiener filter denoiser.
        
        Args:
            frame_size: STFT frame/FFT size
            hop_size: STFT hop size (None = frame_size // 4)
            noise_frames: Number of initial frames for noise estimation
            alpha: Oversubtraction factor (higher = more aggressive)
            beta: Spectral floor to prevent musical noise (0-1)
            noise_psd: Pre-computed noise PSD (None = estimate from signal)
        """
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else frame_size // 4
        self.noise_frames = noise_frames
        self.alpha = alpha
        self.beta = beta
        self.noise_psd = noise_psd
        
        # Create analysis/synthesis window (Hann)
        self.window = signal.windows.hann(frame_size, sym=False)
    
    def _stft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform.
        
        Returns:
            magnitude: Magnitude spectrogram
            phase: Phase spectrogram
        """
        # Pad signal for complete frames
        pad_len = (self.frame_size - len(audio) % self.hop_size) % self.hop_size
        audio_padded = np.pad(audio, (0, pad_len))
        
        # Compute STFT
        f, t, Zxx = signal.stft(
            audio_padded,
            nperseg=self.frame_size,
            noverlap=self.frame_size - self.hop_size,
            window=self.window,
            return_onesided=True
        )
        
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        return magnitude, phase
    
    def _istft(self, magnitude: np.ndarray, phase: np.ndarray,
               original_length: int) -> np.ndarray:
        """
        Compute Inverse Short-Time Fourier Transform.
        
        Args:
            magnitude: Magnitude spectrogram
            phase: Phase spectrogram
            original_length: Original signal length
        
        Returns:
            Reconstructed time-domain signal
        """
        # Reconstruct complex spectrogram
        Zxx = magnitude * np.exp(1j * phase)
        
        # Inverse STFT
        _, audio = signal.istft(
            Zxx,
            nperseg=self.frame_size,
            noverlap=self.frame_size - self.hop_size,
            window=self.window
        )
        
        # Trim to original length
        audio = audio[:original_length]
        
        return audio.astype(np.float32)
    
    def _estimate_noise_psd(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Estimate noise power spectral density from initial frames.
        
        Assumes the first few frames contain mostly noise (or a
        representative noise sample).
        
        Args:
            magnitude: Full magnitude spectrogram
        
        Returns:
            Estimated noise PSD (frequency bins,)
        """
        # Use initial frames for noise estimation
        noise_mag = magnitude[:, :self.noise_frames]
        
        # Estimate noise PSD as average power
        noise_psd = np.mean(noise_mag ** 2, axis=1)
        
        return noise_psd
    
    def _compute_wiener_gain(
        self,
        signal_psd: np.ndarray,
        noise_psd: np.ndarray
    ) -> np.ndarray:
        """
        Compute Wiener filter gain.
        
        Args:
            signal_psd: Noisy signal PSD
            noise_psd: Noise PSD estimate
        
        Returns:
            Wiener gain (0 to 1)
        """
        # Estimate clean signal PSD (spectral subtraction)
        clean_psd_est = np.maximum(signal_psd - self.alpha * noise_psd, 0)
        
        # Wiener gain: H = S_xx / (S_xx + S_nn)
        # With spectral floor to prevent musical noise
        gain = clean_psd_est / (clean_psd_est + noise_psd + 1e-10)
        
        # Apply spectral floor
        gain = np.maximum(gain, self.beta)
        
        return gain
    
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Denoise audio signal using Wiener filtering.
        
        Args:
            audio: Input noisy audio (1D numpy array, float32)
        
        Returns:
            Denoised audio (same shape as input)
        """
        original_length = len(audio)
        
        # Compute STFT
        magnitude, phase = self._stft(audio)
        
        # Estimate or use provided noise PSD
        if self.noise_psd is None:
            noise_psd = self._estimate_noise_psd(magnitude)
        else:
            noise_psd = self.noise_psd
        
        # Reshape noise PSD for broadcasting
        noise_psd = noise_psd[:, np.newaxis]
        
        # Compute noisy signal PSD
        signal_psd = magnitude ** 2
        
        # Compute Wiener gain for each frame
        gain = self._compute_wiener_gain(signal_psd, noise_psd)
        
        # Apply gain to magnitude
        denoised_magnitude = magnitude * gain
        
        # Reconstruct signal
        denoised = self._istft(denoised_magnitude, phase, original_length)
        
        return denoised
    
    def denoise_with_noise_profile(
        self,
        audio: np.ndarray,
        noise_sample: np.ndarray
    ) -> np.ndarray:
        """
        Denoise using a separate noise sample for PSD estimation.
        
        Args:
            audio: Input noisy audio
            noise_sample: Pure noise sample for PSD estimation
        
        Returns:
            Denoised audio
        """
        # Estimate noise PSD from sample
        noise_mag, _ = self._stft(noise_sample)
        self.noise_psd = np.mean(noise_mag ** 2, axis=1)
        
        return self.denoise(audio)
    
    def set_noise_psd(self, noise_psd: np.ndarray):
        """
        Set pre-computed noise PSD.
        
        Args:
            noise_psd: Noise power spectral density
        """
        self.noise_psd = noise_psd
    
    def update_noise_estimate(self, magnitude_frame: np.ndarray, alpha: float = 0.98):
        """
        Update noise estimate adaptively (for real-time use).
        
        Uses exponential smoothing for online noise tracking.
        
        Args:
            magnitude_frame: Current frame magnitude
            alpha: Smoothing factor (0.9-0.99)
        """
        frame_psd = magnitude_frame ** 2
        
        if self.noise_psd is None:
            self.noise_psd = frame_psd
        else:
            self.noise_psd = alpha * self.noise_psd + (1 - alpha) * frame_psd
    
    @property
    def name(self) -> str:
        """Return descriptive name for UI/logging."""
        return f"Wiener (α={self.alpha}, β={self.beta})"
    
    @property
    def category(self) -> str:
        """Denoiser category for UI grouping."""
        return 'classical'


class SpectralSubtraction:
    """
    Spectral Subtraction denoiser (simpler variant of Wiener).
    
    Subtracts estimated noise spectrum from noisy signal spectrum.
    Simpler than Wiener but can introduce musical noise.
    """
    
    def __init__(
        self,
        frame_size: int = 2048,
        hop_size: Optional[int] = None,
        noise_frames: int = 10,
        alpha: float = 2.0,
        beta: float = 0.01
    ):
        """
        Initialize spectral subtraction denoiser.
        
        Args:
            frame_size: STFT frame size
            hop_size: STFT hop size
            noise_frames: Frames for noise estimation
            alpha: Oversubtraction factor
            beta: Spectral floor
        """
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else frame_size // 4
        self.noise_frames = noise_frames
        self.alpha = alpha
        self.beta = beta
        self.window = signal.windows.hann(frame_size, sym=False)
    
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Denoise using spectral subtraction.
        
        Args:
            audio: Input noisy audio
        
        Returns:
            Denoised audio
        """
        original_length = len(audio)
        
        # STFT
        f, t, Zxx = signal.stft(
            audio,
            nperseg=self.frame_size,
            noverlap=self.frame_size - self.hop_size,
            window=self.window
        )
        
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Estimate noise from initial frames
        noise_mag = np.mean(magnitude[:, :self.noise_frames] ** 2, axis=1, keepdims=True)
        
        # Spectral subtraction with floor
        power = magnitude ** 2
        clean_power = np.maximum(power - self.alpha * noise_mag, self.beta * power)
        clean_magnitude = np.sqrt(clean_power)
        
        # Reconstruct
        Zxx_clean = clean_magnitude * np.exp(1j * phase)
        _, denoised = signal.istft(
            Zxx_clean,
            nperseg=self.frame_size,
            noverlap=self.frame_size - self.hop_size,
            window=self.window
        )
        
        return denoised[:original_length].astype(np.float32)
    
    @property
    def name(self) -> str:
        return f"Spectral Sub (α={self.alpha})"
    
    @property
    def category(self) -> str:
        """Denoiser category for UI grouping."""
        return 'classical'


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("Wiener Filter Denoiser Test")
    print("=" * 60)
    
    # Generate test signal
    np.random.seed(42)
    sr = 44100
    t = np.linspace(0, 1, sr)
    clean = np.sin(2 * np.pi * 440 * t) * 0.5
    noise = np.random.randn(len(t)) * 0.1
    noisy = (clean + noise).astype(np.float32)
    
    # Calculate SNR
    def snr(clean, noisy):
        noise = noisy - clean
        return 10 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2))
    
    print(f"Input SNR:  {snr(clean, noisy):.2f} dB")
    
    # Test Wiener filter
    wiener = WienerDenoiser(frame_size=2048, noise_frames=5)
    denoised_wiener = wiener.denoise(noisy)
    print(f"Wiener SNR: {snr(clean, denoised_wiener):.2f} dB")
    
    # Test Spectral Subtraction
    ss = SpectralSubtraction(frame_size=2048, noise_frames=5)
    denoised_ss = ss.denoise(noisy)
    print(f"SS SNR:     {snr(clean, denoised_ss):.2f} dB")
    
    print("\n✅ Wiener filter working!")
