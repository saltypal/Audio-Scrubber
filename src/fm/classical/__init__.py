"""
Denoiser Base Interface and Registry
=====================================
Unified interface for all denoisers (DL and classical) to enable
easy comparison and switching in the UI.

Usage:
    from src.fm.classical import DenoiserRegistry
    
    # List available denoisers
    DenoiserRegistry.list_denoisers()
    
    # Get a denoiser by name
    denoiser = DenoiserRegistry.get('wavelet_soft')
    clean = denoiser.denoise(noisy)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import time
import psutil
import os


@dataclass
class DenoiseResult:
    """Result from a denoiser with metrics."""
    audio: np.ndarray
    processing_time_ms: float
    cpu_percent: float
    memory_mb: float
    snr_improvement: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'processing_time_ms': self.processing_time_ms,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'snr_improvement': self.snr_improvement
        }


class BaseDenoiser(ABC):
    """
    Abstract base class for all denoisers.
    
    All denoisers (DL-based and classical) should implement this interface
    for unified handling in the UI and comparison tools.
    """
    
    @abstractmethod
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        """
        Denoise audio signal.
        
        Args:
            audio: Input noisy audio (1D numpy array, float32)
        
        Returns:
            Denoised audio (same shape as input)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for UI display."""
        pass
    
    @property
    def category(self) -> str:
        """Category: 'classical' or 'deep_learning'."""
        return 'classical'
    
    def denoise_with_metrics(self, audio: np.ndarray,
                             reference: Optional[np.ndarray] = None) -> DenoiseResult:
        """
        Denoise audio and collect performance metrics.
        
        Args:
            audio: Input noisy audio
            reference: Optional clean reference for SNR calculation
        
        Returns:
            DenoiseResult with audio and metrics
        """
        process = psutil.Process(os.getpid())
        
        # Measure CPU and memory before
        cpu_before = process.cpu_percent()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # Time the denoising
        start_time = time.perf_counter()
        denoised = self.denoise(audio)
        end_time = time.perf_counter()
        
        # Measure after
        cpu_after = process.cpu_percent()
        mem_after = process.memory_info().rss / 1024 / 1024
        
        processing_time_ms = (end_time - start_time) * 1000
        cpu_percent = max(cpu_before, cpu_after)
        memory_mb = mem_after - mem_before
        
        # Calculate SNR improvement if reference provided
        snr_improvement = None
        if reference is not None:
            snr_before = self._calculate_snr(reference, audio)
            snr_after = self._calculate_snr(reference, denoised)
            snr_improvement = snr_after - snr_before
        
        return DenoiseResult(
            audio=denoised,
            processing_time_ms=processing_time_ms,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            snr_improvement=snr_improvement
        )
    
    @staticmethod
    def _calculate_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
        """Calculate SNR in dB."""
        noise = noisy - clean
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power < 1e-10:
            return 100.0  # Essentially noise-free
        return 10 * np.log10(signal_power / noise_power)


class PassthroughDenoiser(BaseDenoiser):
    """Passthrough (no processing) for baseline comparison."""
    
    def denoise(self, audio: np.ndarray) -> np.ndarray:
        return audio.copy()
    
    @property
    def name(self) -> str:
        return "Passthrough (No Processing)"
    
    @property
    def category(self) -> str:
        return "baseline"


class DenoiserRegistry:
    """
    Registry of all available denoisers.
    
    Allows dynamic discovery and instantiation of denoisers for the UI.
    """
    
    _denoisers: Dict[str, type] = {}
    _instances: Dict[str, BaseDenoiser] = {}
    
    @classmethod
    def register(cls, key: str, denoiser_class: type, **default_kwargs):
        """
        Register a denoiser class.
        
        Args:
            key: Unique identifier
            denoiser_class: Class implementing BaseDenoiser
            **default_kwargs: Default constructor arguments
        """
        cls._denoisers[key] = (denoiser_class, default_kwargs)
    
    @classmethod
    def get(cls, key: str, **kwargs) -> BaseDenoiser:
        """
        Get or create a denoiser instance.
        
        Args:
            key: Denoiser identifier
            **kwargs: Override default constructor arguments
        
        Returns:
            Denoiser instance
        """
        if key not in cls._denoisers:
            raise KeyError(f"Unknown denoiser: {key}. Available: {list(cls._denoisers.keys())}")
        
        denoiser_class, default_kwargs = cls._denoisers[key]
        merged_kwargs = {**default_kwargs, **kwargs}
        
        # Create new instance
        return denoiser_class(**merged_kwargs)
    
    @classmethod
    def list_denoisers(cls) -> List[Dict[str, str]]:
        """List all registered denoisers with metadata."""
        result = []
        for key, (denoiser_class, kwargs) in cls._denoisers.items():
            # Create temporary instance for metadata
            try:
                instance = denoiser_class(**kwargs)
                result.append({
                    'key': key,
                    'name': instance.name,
                    'category': instance.category
                })
            except Exception:
                result.append({
                    'key': key,
                    'name': key,
                    'category': 'unknown'
                })
        return result
    
    @classmethod
    def get_by_category(cls, category: str) -> List[str]:
        """Get denoiser keys by category."""
        return [
            info['key'] for info in cls.list_denoisers()
            if info['category'] == category
        ]


# Register built-in denoisers
def _register_builtins():
    """Register all built-in denoisers."""
    
    # Baseline
    DenoiserRegistry.register('passthrough', PassthroughDenoiser)
    
    # Classical - Wavelet variants
    try:
        from .wavelet_denoiser import WaveletDenoiser
        
        DenoiserRegistry.register('wavelet_db4_soft', WaveletDenoiser,
                                  wavelet='db4', threshold_mode='soft')
        DenoiserRegistry.register('wavelet_db4_hard', WaveletDenoiser,
                                  wavelet='db4', threshold_mode='hard')
        DenoiserRegistry.register('wavelet_sym8_soft', WaveletDenoiser,
                                  wavelet='sym8', threshold_mode='soft')
        DenoiserRegistry.register('wavelet_coif3_soft', WaveletDenoiser,
                                  wavelet='coif3', threshold_mode='soft')
    except ImportError:
        pass
    
    # Classical - Wiener variants
    try:
        from .wiener_denoiser import WienerDenoiser, SpectralSubtraction
        
        DenoiserRegistry.register('wiener_standard', WienerDenoiser,
                                  alpha=1.0, beta=0.02)
        DenoiserRegistry.register('wiener_aggressive', WienerDenoiser,
                                  alpha=2.0, beta=0.01)
        DenoiserRegistry.register('spectral_subtraction', SpectralSubtraction,
                                  alpha=2.0, beta=0.01)
    except ImportError:
        pass


# Initialize on import
_register_builtins()


# Convenience exports
__all__ = [
    'BaseDenoiser',
    'DenoiseResult', 
    'DenoiserRegistry',
    'PassthroughDenoiser'
]
