"""
FM Monitor: Real-time FM radio receiver with frequency tuning and continuous sampling.
Stream live FM radio to speakers with ability to change frequency on the fly.

Usage:
    python src/fm_monitor.py --frequency 99.5e6
    # Shows frequency, signal strength, and plays to speakers
    
    Commands:
    - Type 'f <freq>' to change frequency (e.g., 'f 101.1')
    - Type 'g <gain>' to change gain (e.g., 'g 40')
    - Type 'q' to quit
"""
import asyncio
import numpy as np
import sounddevice as sd
from threading import Thread, Event
from queue import Queue, Empty
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from src.fm.fm_record.rtlsdr_core import RTLSDRCore
from config import AudioSettings


class FMMonitor:
    """Real-time FM radio monitor with interactive tuning."""
    
    def __init__(self, frequency, gain='auto'):
        self.frequency = frequency
        self.gain = gain
        self.core = RTLSDRCore(
            frequency=frequency,
            sample_rate=AudioSettings.SAMPLE_RATE,
            chunk_size=AudioSettings.CHUNK_SIZE,
            gain=gain
        )
        
        self.audio_queue = Queue(maxsize=10)
        self.running = Event()
        self.tune_event = asyncio.Event()
        self.new_frequency = None
    
    def start(self):
        """Start FM monitoring."""
        print(f"\n{'='*60}")
        print("FM Monitor - Real-Time Radio Receiver")
        print(f"{'='*60}\n")
        
        self.running.set()
        
        # Start audio streaming thread
        sdr_thread = Thread(target=self._sdr_thread, daemon=True)
        audio_thread = Thread(target=self._audio_thread, daemon=True)
        
        sdr_thread.start()
        audio_thread.start()
        
        print(f"📻 Listening to {self.frequency/1e6:.2f} MHz")
        print("Commands: f <freq> (tune), g <gain> (gain), q (quit)\n")
        
        # Interactive command loop
        try:
            while self.running.is_set():
                try:
                    cmd = input().strip()
                    if not cmd:
                        continue
                    
                    if cmd.lower() == 'q':
                        print("Stopping...")
                        self.running.clear()
                        break
                    
                    elif cmd.lower().startswith('f '):
                        try:
                            freq_mhz = float(cmd.split()[1])
                            self.frequency = freq_mhz * 1e6
                            self.core.tune(self.frequency)
                        except (ValueError, IndexError):
                            print("Usage: f <frequency_in_MHz> (e.g., f 101.1)")
                    
                    elif cmd.lower().startswith('g '):
                        try:
                            gain = float(cmd.split()[1])
                            self.core.set_gain(gain)
                        except (ValueError, IndexError):
                            print("Usage: g <gain_in_dB> (e.g., g 40)")
                    
                    else:
                        print("Unknown command. Use: f <freq>, g <gain>, q")
                
                except EOFError:
                    break
        
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self.running.clear()
    
    def _sdr_thread(self):
        """SDR capture thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._sdr_async())
        finally:
            loop.close()
    
    async def _sdr_async(self):
        """Async SDR streaming."""
        await self.core.initialize()
        
        try:
            async for audio_chunk in self.core.stream_audio():
                if not self.running.is_set():
                    break
                self.audio_queue.put(audio_chunk)
        
        except Exception as e:
            print(f"❌ SDR error: {e}")
        
        finally:
            await self.core.close()
    
    def _audio_thread(self):
        """Audio playback thread."""
        def audio_callback(outdata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            
            try:
                chunk = self.audio_queue.get_nowait()
                outdata[:, 0] = chunk
            except Empty:
                outdata.fill(0)
        
        try:
            with sd.OutputStream(
                samplerate=AudioSettings.SAMPLE_RATE,
                blocksize=AudioSettings.CHUNK_SIZE,
                channels=1,
                dtype='float32',
                callback=audio_callback
            ):
                while self.running.is_set():
                    sd.sleep(100)
        
        except Exception as e:
            print(f"❌ Audio error: {e}")
        
        finally:
            print("✅ Audio stream closed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-time FM radio monitor")
    parser.add_argument('--frequency', type=float, default=99.5e6,
                        help='Starting frequency in Hz (default: 99.5 MHz)')
    parser.add_argument('--gain', default='auto',
                        help="SDR gain ('auto' or dB value)")
    
    args = parser.parse_args()
    
    monitor = FMMonitor(frequency=args.frequency, gain=args.gain)
    monitor.start()


if __name__ == "__main__":
    main()
