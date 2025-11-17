"""
RTL-FM Wrapper: Use the rtl_fm binary for recording and monitoring.
Alternative to pyrtlsdr for when you prefer the C-based rtl_fm tool.

Usage:
    # Record audio from 99.5 MHz for 60 seconds
    python src/rtlfm_wrapper.py record --frequency 99.5e6 --duration 60 --output dataset/noise/
    
    # Monitor FM radio live (stream to speakers)
    python src/rtlfm_wrapper.py monitor --frequency 99.5e6
"""
import subprocess
import numpy as np
import soundfile as sf
import sounddevice as sd
from pathlib import Path
import argparse
import sys
import time
from threading import Thread, Event
from queue import Queue, Empty
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Paths, AudioSettings


class RTLFMWrapper:
    """Wrapper around rtl_fm binary for recording and monitoring."""
    
    def __init__(self, frequency, sample_rate=AudioSettings.SAMPLE_RATE, gain='auto'):
        """
        Initialize RTL-FM wrapper.
        
        Args:
            frequency: Frequency in Hz
            sample_rate: Output sample rate (default: 22050 Hz)
            gain: SDR gain ('auto' or dB value)
        """
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.gain = gain
        self.process = None
        
        # Check if rtl_fm exists
        try:
            result = subprocess.run(['rtl_fm.exe', '--help'], 
                                  capture_output=True, timeout=2)
            print("✅ rtl_fm.exe found")
        except FileNotFoundError:
            print("❌ rtl_fm.exe not found in PATH")
            print("   Install from: https://osmocom.org/projects/rtl-sdr/wiki/Rtl-sdr")
            raise
    
    def _build_rtl_fm_command(self, duration=None):
        """Build rtl_fm command."""
        cmd = [
            'rtl_fm.exe',
            '-f', str(int(self.frequency)),  # Frequency
            '-M', 'wbfm',                     # Wideband FM
            '-s', '200k',                     # SDR sample rate
            '-r', str(self.sample_rate),      # Output sample rate
            '-g', str(self.gain) if self.gain != 'auto' else '40',  # Gain
            '-'                               # Output to stdout
        ]
        return cmd
    
    def record(self, duration, output_dir):
        """
        Record audio to file using rtl_fm.
        
        Args:
            duration: Recording duration in seconds
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"audio_{self.frequency/1e6:.1f}MHz_{timestamp}.wav"
        
        print(f"\n{'='*60}")
        print("RTL-FM Recorder (using rtl_fm binary)")
        print(f"{'='*60}")
        print(f"Frequency: {self.frequency/1e6:.2f} MHz")
        print(f"Duration: {duration} seconds")
        print(f"Output: {output_file}")
        print(f"{'='*60}\n")
        
        cmd = self._build_rtl_fm_command(duration)
        
        try:
            print("⏹️  Recording...")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=4096
            )
            
            audio_data = []
            bytes_per_second = self.sample_rate * 2  # 16-bit = 2 bytes
            bytes_to_read = bytes_per_second * duration
            bytes_read = 0
            start_time = time.time()
            
            while bytes_read < bytes_to_read:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    break
                
                audio_data.append(chunk)
                bytes_read += len(chunk)
                
                elapsed = time.time() - start_time
                progress = (bytes_read / bytes_to_read) * 100
                print(f"\r  {progress:.1f}% | {elapsed:.1f}s", end='', flush=True)
            
            # Terminate process
            self.process.terminate()
            self.process.wait(timeout=2)
            
            print("\n✅ Recording complete!")
            
            # Convert to audio
            audio_int16 = np.frombuffer(b''.join(audio_data), dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            sf.write(str(output_file), audio_float32, self.sample_rate)
            print(f"💾 Saved: {output_file.name} ({len(audio_float32)/self.sample_rate:.1f}s)")
        
        except KeyboardInterrupt:
            print("\nRecording cancelled")
            if self.process:
                self.process.terminate()
        
        except Exception as e:
            print(f"❌ Error: {e}")
            if self.process:
                self.process.terminate()
    
    def monitor(self):
        """Monitor FM radio live with playback."""
        print(f"\n{'='*60}")
        print("RTL-FM Monitor (using rtl_fm binary)")
        print(f"{'='*60}")
        print(f"Frequency: {self.frequency/1e6:.2f} MHz")
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"{'='*60}\n")
        
        cmd = self._build_rtl_fm_command()
        audio_queue = Queue(maxsize=10)
        running = Event()
        running.set()
        
        def capture_thread():
            """Read audio from rtl_fm."""
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=8192
                )
                
                chunk_size = self.sample_rate // 10  # 100ms chunks
                bytes_per_chunk = chunk_size * 2  # 16-bit
                
                print(f"📻 Streaming from rtl_fm...")
                
                while running.is_set():
                    chunk = self.process.stdout.read(bytes_per_chunk)
                    if not chunk:
                        break
                    
                    audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    
                    audio_queue.put(audio_float32)
            
            except Exception as e:
                print(f"❌ Capture error: {e}")
            
            finally:
                if self.process:
                    self.process.terminate()
        
        def playback_thread():
            """Play audio to speakers."""
            try:
                def audio_callback(outdata, frames, time_info, status):
                    if status:
                        print(f"Audio status: {status}")
                    
                    try:
                        chunk = audio_queue.get_nowait()
                        if len(chunk) >= frames:
                            outdata[:, 0] = chunk[:frames]
                        else:
                            outdata[:len(chunk), 0] = chunk
                            outdata[len(chunk):, 0] = 0
                    except Empty:
                        outdata.fill(0)
                
                with sd.OutputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.sample_rate // 10,
                    channels=1,
                    dtype='float32',
                    callback=audio_callback
                ):
                    print("🔊 Audio playback started")
                    while running.is_set():
                        sd.sleep(100)
            
            except Exception as e:
                print(f"❌ Playback error: {e}")
        
        # Start threads
        capture = Thread(target=capture_thread, daemon=True)
        playback = Thread(target=playback_thread, daemon=True)
        
        capture.start()
        playback.start()
        
        try:
            while running.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            running.clear()
            capture.join(timeout=2)
            playback.join(timeout=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RTL-FM wrapper for recording/monitoring")
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Record subcommand
    record_parser = subparsers.add_parser('record', help='Record audio')
    record_parser.add_argument('--frequency', type=float, default=99.5e6,
                               help='Frequency in Hz (default: 99.5 MHz)')
    record_parser.add_argument('--duration', type=int, default=10,
                               help='Duration in seconds (default: 10)')
    record_parser.add_argument('--output', type=str, default=str(Paths.NOISE_ROOT),
                               help='Output directory')
    record_parser.add_argument('--gain', default='auto',
                               help="SDR gain ('auto' or dB value)")
    
    # Monitor subcommand
    monitor_parser = subparsers.add_parser('monitor', help='Monitor FM radio')
    monitor_parser.add_argument('--frequency', type=float, default=99.5e6,
                                help='Frequency in Hz (default: 99.5 MHz)')
    monitor_parser.add_argument('--gain', default='auto',
                                help="SDR gain ('auto' or dB value)")
    
    args = parser.parse_args()
    
    if args.command == 'record':
        wrapper = RTLFMWrapper(frequency=args.frequency, gain=args.gain)
        wrapper.record(duration=args.duration, output_dir=args.output)
    
    elif args.command == 'monitor':
        wrapper = RTLFMWrapper(frequency=args.frequency, gain=args.gain)
        wrapper.monitor()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
