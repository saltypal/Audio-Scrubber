import torch
import numpy as np
import sounddevice as sd
from queue import Queue
from threading import Thread, Event
from model.neuralnet import UNet1D
from pathlib import Path
import soundfile as sf
import time

"""
Created by Satya with Copilot @ 15/11/25

Real-Time Audio Denoiser using Multi-Threaded Architecture
- Audio Thread: Captures mic input and plays denoised output (high priority)
- AI Thread: Processes audio chunks through the U-Net model
- Uses queues as "conveyor belts" to prevent stuttering
- Pure real-time: no recording, direct mic-to-speaker with denoising
"""

class RealTimeDenoiser:
    """
    Real-time audio denoiser with multi-threaded queue architecture.
    """
    def __init__(self, model_path, chunk_size=4096, sample_rate=22050, device='cpu'):
        """
        Initialize the real-time denoiser.
        
        Args:
            model_path: Path to trained model checkpoint
            chunk_size: Size of audio chunks (smaller = lower latency, but more overhead)
            sample_rate: Audio sample rate
            device: 'cuda' or 'cpu'
        """
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = UNet1D(in_channels=1, out_channels=1).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✅ Model loaded! (Val Loss: {checkpoint['val_loss']:.6f})")
        
        # Queues (conveyor belts)
        self.in_queue = Queue(maxsize=10)   # Mic audio goes here
        self.out_queue = Queue(maxsize=10)  # Clean audio comes from here
        
        # Control flags
        self.running = Event()
        self.ai_thread = None
        
        # For tracking performance
        self.processed_chunks = 0
        
        print(f"Configuration:")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  Chunk Size: {chunk_size} samples (~{chunk_size/sample_rate*1000:.1f} ms)")
        print(f"  Device: {device}")
    
    def _ai_worker(self):
        """
        AI Thread: Continuously processes audio chunks through the model.
        This runs in a separate thread so it never blocks the audio callback.
        """
        print("🤖 AI thread started")
        
        with torch.no_grad():  # No gradients needed for inference
            while self.running.is_set():
                # Get noisy audio from input queue (blocks until available)
                if not self.in_queue.empty():
                    noisy_chunk = self.in_queue.get()
                    
                    # Convert to tensor and add batch + channel dimensions
                    # Shape: (1, 1, chunk_size)
                    noisy_tensor = torch.FloatTensor(noisy_chunk).unsqueeze(0).unsqueeze(0)
                    noisy_tensor = noisy_tensor.to(self.device)
                    
                    # Run through model
                    clean_tensor = self.model(noisy_tensor)
                    
                    # Convert back to numpy
                    clean_chunk = clean_tensor.squeeze().cpu().numpy()
                    
                    # Put clean audio in output queue
                    if not self.out_queue.full():
                        self.out_queue.put(clean_chunk)
                        self.processed_chunks += 1
                    else:
                        # Queue full - skip this chunk to avoid blocking
                        pass
        
        print("🤖 AI thread stopped")
    
    def _audio_callback(self, indata, outdata, frames, time, status):
        """
        Audio Thread: Called by sounddevice for each audio chunk.
        This must be FAST - no heavy processing here!
        
        Args:
            indata: Input audio from microphone (numpy array)
            outdata: Output audio to speakers (numpy array, we fill this)
            frames: Number of frames (should equal chunk_size)
            time: Timing information
            status: Status flags
        """
        if status:
            print(f"⚠️ Audio status: {status}")
        
        # Step 1: Grab mic input and put in AI queue (non-blocking)
        noisy_chunk = indata[:, 0].copy()  # Take first channel (mono)
        
        if not self.in_queue.full():
            self.in_queue.put(noisy_chunk)
        
        # Step 2: Try to get clean audio from AI queue
        if not self.out_queue.empty():
            clean_chunk = self.out_queue.get()
            outdata[:, 0] = clean_chunk
        else:
            # No clean audio ready yet - output silence (prevents crackling)
            outdata.fill(0)
    
    def start(self):
        """
        Start pure real-time denoising.
        Opens audio stream (mic input -> denoised output to speakers).
        Runs continuously until interrupted.
        """
        print("\n" + "="*60)
        print("Starting Pure Real-Time Audio Denoiser")
        print("="*60)
        print("🎤 Mic input -> 🤖 AI Denoising -> 🔊 Speaker output")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Set running flag
        self.running.set()
        
        # Start AI worker thread
        self.ai_thread = Thread(target=self._ai_worker, daemon=True)
        self.ai_thread.start()
        
        # Start audio stream (this blocks until stopped)
        try:
            with sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,  # Mono
                dtype='float32',
                callback=self._audio_callback
            ):
                print("✅ Real-time denoising active!")
                print("   Speak into your microphone...")
                print("   Denoised audio will play through speakers\n")
                
                # Keep running until interrupted (infinite loop)
                while self.running.is_set():
                    sd.sleep(100)  # Sleep for 100ms, check again
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping...")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        finally:
            self.stop()
    
    def stop(self):
        """
        Stop real-time denoising.
        """
        print("\nShutting down...")
        
        # Clear running flag
        self.running.clear()
        
        # Wait for AI thread to finish
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=2.0)
        
        print(f"\n📊 Statistics:")
        print(f"   Processed chunks: {self.processed_chunks}")
        print(f"   Total time: ~{self.processed_chunks * self.chunk_size / self.sample_rate:.1f} seconds")
        print("\n✅ Stopped successfully!")
    
    def stop(self):
        """
        Stop real-time denoising.
        """
        print("\nShutting down...")
        
        # Clear running flag
        self.running.clear()
        
        # Wait for AI thread to finish
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=2.0)
        
        print(f"\n📊 Statistics:")
        print(f"   Processed chunks: {self.processed_chunks}")
        print(f"   Total time: ~{self.processed_chunks * self.chunk_size / self.sample_rate:.1f} seconds")
        print("\n✅ Stopped successfully!")


def main():
    """
    Main function to run pure real-time denoising.
    """
    # Configuration
    MODEL_PATH = r"saved_models\unet1d_best.pth"
    CHUNK_SIZE = 4096  # ~186ms latency at 22050 Hz (lower = less latency)
    SAMPLE_RATE = 22050
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please train the model first using backshot.py")
        return
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create denoiser
    denoiser = RealTimeDenoiser(
        model_path=MODEL_PATH,
        chunk_size=CHUNK_SIZE,
        sample_rate=SAMPLE_RATE,
        device=device
    )
    
    # Start pure real-time processing (runs until Ctrl+C)
    denoiser.start()


if __name__ == "__main__":
    # Check if sounddevice is installed
    try:
        import sounddevice as sd
    except ImportError:
        print("❌ sounddevice not installed!")
        print("Install it with: pip install sounddevice")
        exit(1)
    
    main()
