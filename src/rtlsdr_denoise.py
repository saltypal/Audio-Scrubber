import torch
import numpy as np
import sounddevice as sd
from queue import Queue
from threading import Thread, Event
import sys
from pathlib import Path
import subprocess
import struct

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Paths, AudioSettings, RTLSDRSettings
from model.neuralnet import UNet1D

"""
Created by Satya with Copilot @ 15/11/25

Real-Time Audio Denoiser for RTL-SDR FM Radio
- RTL-SDR Thread: Captures FM radio audio via rtl_fm
- AI Thread: Denoises audio chunks through U-Net model
- Audio Thread: Plays denoised audio to speakers
- Uses queues as "conveyor belts" for smooth operation
"""

class RTLSDRDenoiser:
    """
    Real-time audio denoiser for RTL-SDR FM radio input.
    """
    def __init__(self, model_path=None, frequency=None, sample_rate=None, chunk_size=None, device='cpu'):
        """
        Initialize the RTL-SDR denoiser.
        
        Args:
            model_path: Path to trained model checkpoint (default: uses config)
            frequency: FM radio frequency in Hz (default: uses config, e.g., 99.5e6 = 99.5 MHz)
            sample_rate: Audio sample rate (default: uses config)
            chunk_size: Size of audio chunks for processing (default: uses config)
            device: 'cuda' or 'cpu'
        """
        # Use config defaults if not specified
        if model_path is None:
            model_path = str(Paths.MODEL_BEST)
        if frequency is None:
            frequency = RTLSDRSettings.FM_FREQUENCY
        if sample_rate is None:
            sample_rate = AudioSettings.SAMPLE_RATE
        if chunk_size is None:
            chunk_size = AudioSettings.CHUNK_SIZE
        
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = UNet1D(in_channels=1, out_channels=1).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✅ Model loaded! (Val Loss: {checkpoint['val_loss']:.6f})")
        
        # Queues (conveyor belts)
        self.rtl_queue = Queue(maxsize=10)   # Raw FM audio from RTL-SDR
        self.out_queue = Queue(maxsize=10)   # Denoised audio for playback
        
        # Control flags
        self.running = Event()
        self.rtl_thread = None
        self.ai_thread = None
        self.audio_thread = None
        
        # RTL-SDR process
        self.rtl_process = None
        
        # Statistics
        self.processed_chunks = 0
        
        print(f"Configuration:")
        print(f"  FM Frequency: {frequency/1e6:.2f} MHz")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  Chunk Size: {chunk_size} samples (~{chunk_size/sample_rate*1000:.1f} ms)")
        print(f"  Device: {device}")
    
    def _rtl_worker(self):
        """
        RTL-SDR Thread: Captures FM radio audio using rtl_fm.
        Runs rtl_fm subprocess and reads raw audio data.
        """
        print("📻 RTL-SDR thread started")
        
        # Start rtl_fm process
        # rtl_fm demodulates FM and outputs raw 16-bit signed PCM audio
        rtl_cmd = [
            'rtl_fm',
            '-f', str(int(self.frequency)),  # Frequency
            '-M', 'wbfm',                     # Wideband FM (for FM radio)
            '-s', '200k',                     # Sample rate for SDR (200 kHz)
            '-r', str(self.sample_rate),      # Audio output rate
            '-g', '40',                       # RF gain
            '-'                               # Output to stdout
        ]
        
        try:
            self.rtl_process = subprocess.Popen(
                rtl_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.chunk_size * 2  # 2 bytes per sample (16-bit)
            )
            
            print(f"✅ RTL-SDR capturing FM at {self.frequency/1e6:.2f} MHz")
            
            # Read audio data in chunks
            bytes_per_sample = 2  # 16-bit = 2 bytes
            bytes_per_chunk = self.chunk_size * bytes_per_sample
            
            while self.running.is_set():
                # Read raw audio data
                raw_data = self.rtl_process.stdout.read(bytes_per_chunk)
                
                if len(raw_data) < bytes_per_chunk:
                    print("⚠️ RTL-SDR: Incomplete chunk, stopping")
                    break
                
                # Convert 16-bit signed PCM to float32 numpy array
                # Format: '<h' = little-endian signed short (16-bit)
                samples = struct.unpack(f'<{self.chunk_size}h', raw_data)
                audio_chunk = np.array(samples, dtype=np.float32) / 32768.0  # Normalize to [-1, 1]
                
                # Put in queue for AI processing
                if not self.rtl_queue.full():
                    self.rtl_queue.put(audio_chunk)
        
        except FileNotFoundError:
            print("❌ rtl_fm not found! Please install rtl-sdr:")
            print("   sudo apt install rtl-sdr  (Linux)")
            print("   Or download from: https://osmocom.org/projects/rtl-sdr/wiki/Rtl-sdr")
        
        except Exception as e:
            print(f"❌ RTL-SDR error: {e}")
        
        finally:
            if self.rtl_process:
                self.rtl_process.terminate()
                self.rtl_process.wait()
            print("📻 RTL-SDR thread stopped")
    
    def _ai_worker(self):
        """
        AI Thread: Continuously processes audio chunks through the model.
        """
        print("🤖 AI thread started")
        
        with torch.no_grad():
            while self.running.is_set():
                if not self.rtl_queue.empty():
                    noisy_chunk = self.rtl_queue.get()
                    
                    # Convert to tensor (1, 1, chunk_size)
                    noisy_tensor = torch.FloatTensor(noisy_chunk).unsqueeze(0).unsqueeze(0)
                    noisy_tensor = noisy_tensor.to(self.device)
                    
                    # Run through model
                    clean_tensor = self.model(noisy_tensor)
                    
                    # Convert back to numpy
                    clean_chunk = clean_tensor.squeeze().cpu().numpy()
                    
                    # Put in output queue
                    if not self.out_queue.full():
                        self.out_queue.put(clean_chunk)
                        self.processed_chunks += 1
        
        print("🤖 AI thread stopped")
    
    def _audio_worker(self):
        """
        Audio Thread: Plays denoised audio to speakers.
        """
        print("🔊 Audio playback thread started")
        
        def audio_callback(outdata, frames, time, status):
            if status:
                print(f"⚠️ Audio status: {status}")
            
            # Get clean audio from queue
            if not self.out_queue.empty():
                clean_chunk = self.out_queue.get()
                outdata[:, 0] = clean_chunk
            else:
                # No audio ready - output silence
                outdata.fill(0)
        
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype='float32',
                callback=audio_callback
            ):
                while self.running.is_set():
                    sd.sleep(100)
        
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
        
        print("🔊 Audio playback thread stopped")
    
    def start(self):
        """
        Start real-time RTL-SDR denoising.
        """
        print("\n" + "="*60)
        print("Starting RTL-SDR Real-Time Audio Denoiser")
        print("="*60)
        print("📻 RTL-SDR FM -> 🤖 AI Denoising -> 🔊 Speaker output")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Set running flag
        self.running.set()
        
        # Start all threads
        self.rtl_thread = Thread(target=self._rtl_worker, daemon=True)
        self.ai_thread = Thread(target=self._ai_worker, daemon=True)
        self.audio_thread = Thread(target=self._audio_worker, daemon=True)
        
        self.rtl_thread.start()
        self.ai_thread.start()
        self.audio_thread.start()
        
        print("✅ All threads running!")
        print(f"   Listening to FM {self.frequency/1e6:.2f} MHz")
        print("   Denoised audio will play through speakers\n")
        
        try:
            # Keep running until interrupted
            while self.running.is_set():
                sd.sleep(100)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping...")
        
        finally:
            self.stop()
    
    def stop(self):
        """
        Stop real-time denoising.
        """
        print("\nShutting down...")
        
        # Clear running flag
        self.running.clear()
        
        # Wait for threads to finish
        if self.rtl_thread and self.rtl_thread.is_alive():
            self.rtl_thread.join(timeout=2.0)
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=2.0)
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        print(f"\n📊 Statistics:")
        print(f"   Processed chunks: {self.processed_chunks}")
        print(f"   Total time: ~{self.processed_chunks * self.chunk_size / self.sample_rate:.1f} seconds")
        print("\n✅ Stopped successfully!")


def main():
    """
    Main function to run RTL-SDR real-time denoising.
    """
    # Configuration
    MODEL_PATH = r"saved_models\unet1d_best.pth"
    FM_FREQUENCY = 88.5e6  # 88.5 MHz - Change this to your desired FM station
    SAMPLE_RATE = 22050
    CHUNK_SIZE = 4096
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please train the model first using backshot.py")
        return
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print("RTL-SDR Real-Time Denoiser Setup")
    print(f"{'='*60}")
    print(f"Default FM Frequency: {FM_FREQUENCY/1e6:.2f} MHz")
    print("\nYou can change the frequency by editing FM_FREQUENCY in main()")
    print("Example frequencies:")
    print("  88.5 MHz  ->  FM_FREQUENCY = 88.5e6")
    print("  101.1 MHz ->  FM_FREQUENCY = 101.1e6")
    print(f"{'='*60}\n")
    
    # Create denoiser
    denoiser = RTLSDRDenoiser(
        model_path=MODEL_PATH,
        frequency=FM_FREQUENCY,
        sample_rate=SAMPLE_RATE,
        chunk_size=CHUNK_SIZE,
        device=device
    )
    
    # Start real-time processing
    denoiser.start()


if __name__ == "__main__":
    main()
