"""
RTLSDRCore: Core RTL-SDR functionality abstracted into a reusable class.
Handles all aspects of SDR connection, FM demodulation, and async streaming.
"""
import numpy as np
import asyncio
from scipy import signal
from pathlib import Path
import sys


class RTLSDRCore:
    """
    Core RTL-SDR functionality for capturing and demodulating FM radio.
    
    Usage:
        core = RTLSDRCore(frequency=88.5e6, sample_rate=22050)
        core.start()  # Starts async streaming
        for audio_chunk in core.get_audio():
            # Process audio_chunk
    """
    
    def __init__(self, 
                 frequency=99.5e6,
                 sample_rate=22050,
                 sdr_sample_rate=240000,
                 gain='auto',
                 chunk_size=4096):
        """
        Initialize RTL-SDR core.
        
        Args:
            frequency: Target frequency in Hz (default: 99.5 MHz)
            sample_rate: Output audio sample rate (default: 22050 Hz)
            sdr_sample_rate: SDR capture rate (default: 240000 Hz)
            gain: SDR gain ('auto' or dB value, default: 'auto')
            chunk_size: Audio chunk size in samples (default: 4096)
        """
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.sdr_sample_rate = sdr_sample_rate
        self.gain = gain
        self.chunk_size = chunk_size
        self.sdr = None
        
        # Calculated properties
        self.decimation_rate = int(sdr_sample_rate / sample_rate)
        self.samples_per_read = chunk_size * self.decimation_rate
        
    def __repr__(self):
        return (f"RTLSDRCore(freq={self.frequency/1e6:.2f}MHz, "
                f"sr={self.sample_rate}Hz, gain={self.gain})")
    
    async def initialize(self):
        """Initialize RTL-SDR connection using async version."""
        try:
            from rtlsdr.rtlsdraio import RtlSdrAio
            self.sdr = RtlSdrAio()
            await self.sdr.open()
            self.sdr.sample_rate = self.sdr_sample_rate
            self.sdr.center_freq = self.frequency
            self.sdr.gain = self.gain if self.gain != 'auto' else 'auto'
            print(f"✅ SDR initialized: {self}")
            return True
        except Exception as e:
            print(f"❌ SDR init failed: {e}")
            return False
    
    async def close(self):
        """Close SDR connection."""
        if self.sdr:
            try:
                await self.sdr.stop()
                self.sdr.close()
                print("✅ SDR closed")
            except:
                pass
    
    def _demodulate_fm(self, iq_samples):
        """
        Demodulate FM signal using polar discriminator.
        
        Args:
            iq_samples: Complex IQ samples
            
        Returns:
            Demodulated audio (float32 normalized to [-1, 1])
        """
        # Polar discriminator: angle difference between consecutive samples
        angle_diff = np.angle(iq_samples[1:] * np.conj(iq_samples[:-1]))
        
        # Decimation (resampling to target audio rate)
        audio_raw = signal.decimate(angle_diff, self.decimation_rate)
        
        # Normalize angle from [-π, π] to [-1, 1]
        audio_chunk = np.float32(audio_raw[:self.chunk_size]) / np.pi
        
        # Pad if necessary
        if len(audio_chunk) < self.chunk_size:
            padding = np.zeros(self.chunk_size - len(audio_chunk), dtype=np.float32)
            audio_chunk = np.concatenate((audio_chunk, padding))
        
        return audio_chunk
    
    async def stream_audio(self):
        """
        Async generator that yields demodulated FM audio chunks.
        
        Yields:
            audio_chunk: Demodulated audio (float32, shape: (chunk_size,))
        """
        if not self.sdr:
            await self.initialize()
        
        try:
            async for iq_samples in self.sdr.stream(self.samples_per_read):
                audio = self._demodulate_fm(iq_samples)
                yield audio
        except Exception as e:
            print(f"❌ Stream error: {e}")
    
    async def capture_samples(self, duration_seconds):
        """
        Capture raw IQ samples for specified duration.
        
        Args:
            duration_seconds: How long to capture
            
        Yields:
            iq_samples: Raw complex IQ samples
        """
        if not self.sdr:
            await self.initialize()
        
        samples_needed = int(self.sdr_sample_rate * duration_seconds)
        samples_read = 0
        
        try:
            async for iq_samples in self.sdr.stream(self.samples_per_read):
                yield iq_samples
                samples_read += len(iq_samples)
                if samples_read >= samples_needed:
                    break
        except Exception as e:
            print(f"❌ Capture error: {e}")
    
    def tune(self, frequency):
        """Change frequency on the fly."""
        if self.sdr:
            self.sdr.center_freq = frequency
            self.frequency = frequency
            print(f"✅ Tuned to {frequency/1e6:.2f} MHz")
        else:
            print("❌ SDR not initialized")
    
    def set_gain(self, gain):
        """Change gain on the fly."""
        if self.sdr:
            try:
                self.sdr.gain = gain
                self.gain = gain
                print(f"✅ Gain set to {gain}")
            except Exception as e:
                print(f"❌ Gain change failed: {e}")
        else:
            print("❌ SDR not initialized")
