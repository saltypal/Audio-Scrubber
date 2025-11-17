"""
Frequency Recorder: Record IQ samples and demodulated audio from any frequency.
Useful for building noise datasets, capturing specific signals, and analysis.

Usage:
    python src/frequency_recorder.py --frequency 88.5e6 --duration 60 --output dataset/noise/
    # Records 60 seconds of FM at 88.5 MHz and saves to dataset/noise/
"""
import asyncio
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from src.fm.fm_record.rtlsdr_core import RTLSDRCore
from config import Paths, AudioSettings


class FrequencyRecorder:
    """Record IQ samples and/or demodulated audio from RTL-SDR."""
    
    def __init__(self, 
                 frequency,
                 duration,
                 output_dir,
                 record_iq=False,
                 record_audio=True):
        """
        Initialize recorder.
        
        Args:
            frequency: Frequency to record in Hz
            duration: Duration in seconds
            output_dir: Where to save files
            record_iq: Whether to save raw IQ samples
            record_audio: Whether to save demodulated audio
        """
        self.frequency = frequency
        self.duration = duration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_iq = record_iq
        self.record_audio = record_audio
        
        # Timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create RTLSDRCore
        self.core = RTLSDRCore(
            frequency=frequency,
            sample_rate=AudioSettings.SAMPLE_RATE,
            chunk_size=AudioSettings.CHUNK_SIZE
        )
        
        self.audio_chunks = []
        self.iq_samples = []
    
    async def record(self):
        """Start recording."""
        print(f"\n{'='*60}")
        print(f"Frequency Recorder")
        print(f"{'='*60}")
        print(f"Frequency: {self.frequency/1e6:.2f} MHz")
        print(f"Duration: {self.duration} seconds")
        print(f"Output: {self.output_dir}")
        print(f"Recording IQ: {self.record_iq}")
        print(f"Recording Audio: {self.record_audio}")
        print(f"{'='*60}\n")
        
        await self.core.initialize()
        
        print(f"⏹️  Recording started...")
        
        start_time = asyncio.get_event_loop().time()
        
        if self.record_iq:
            # Record raw IQ samples
            async for iq_chunk in self.core.capture_samples(self.duration):
                self.iq_samples.append(iq_chunk)
                elapsed = asyncio.get_event_loop().time() - start_time
                progress = (elapsed / self.duration) * 100
                print(f"\r  IQ: {progress:.1f}% | {len(self.iq_samples)} chunks", end='', flush=True)
                
                if elapsed >= self.duration:
                    break
        
        elif self.record_audio:
            # Record demodulated audio
            start_time = asyncio.get_event_loop().time()
            async for audio_chunk in self.core.stream_audio():
                self.audio_chunks.append(audio_chunk)
                elapsed = asyncio.get_event_loop().time() - start_time
                progress = (elapsed / self.duration) * 100
                print(f"\r  Audio: {progress:.1f}% | {len(self.audio_chunks)} chunks", end='', flush=True)
                
                if elapsed >= self.duration:
                    break
        
        await self.core.close()
        print("\n✅ Recording complete!")
        
        # Save files
        self._save_files()
    
    def _save_files(self):
        """Save recorded data to disk."""
        if self.audio_chunks:
            audio_data = np.concatenate(self.audio_chunks).astype(np.float32)
            audio_path = self.output_dir / f"audio_{self.frequency/1e6:.1f}MHz_{self.timestamp}.wav"
            sf.write(str(audio_path), audio_data, AudioSettings.SAMPLE_RATE)
            print(f"💾 Saved audio: {audio_path.name} ({len(audio_data)/AudioSettings.SAMPLE_RATE:.1f}s)")
        
        if self.iq_samples:
            iq_data = np.concatenate(self.iq_samples).astype(np.complex64)
            iq_path = self.output_dir / f"iq_{self.frequency/1e6:.1f}MHz_{self.timestamp}.npy"
            np.save(str(iq_path), iq_data)
            print(f"💾 Saved IQ: {iq_path.name} ({len(iq_data)} samples)")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Record audio/IQ from any frequency")
    parser.add_argument('--frequency', type=float, default=88.5e6,
                        help='Frequency in Hz (default: 88.5 MHz)')
    parser.add_argument('--duration', type=int, default=10,
                        help='Recording duration in seconds (default: 10)')
    parser.add_argument('--output', type=str, default=str(Paths.NOISE_ROOT),
                        help=f'Output directory (default: {Paths.NOISE_ROOT})')
    parser.add_argument('--iq', action='store_true',
                        help='Record raw IQ samples (larger file size)')
    parser.add_argument('--audio', action='store_true',
                        help='Record demodulated audio (default if neither specified)')
    
    args = parser.parse_args()
    
    # Default to audio if neither specified
    record_iq = args.iq
    record_audio = args.audio or (not args.iq)
    
    recorder = FrequencyRecorder(
        frequency=args.frequency,
        duration=args.duration,
        output_dir=args.output,
        record_iq=record_iq,
        record_audio=record_audio
    )
    
    await recorder.record()


if __name__ == "__main__":
    asyncio.run(main())
