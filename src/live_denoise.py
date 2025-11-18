import torch
import numpy as np
import sounddevice as sd
from queue import Queue
from threading import Thread, Event
import sys
from pathlib import Path
import soundfile as sf
import time

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Paths, AudioSettings
from src.fm.model.neuralnet import UNet1D

"""
================================================================================
LIVE SDR AUDIO DENOISER - VIRTUAL CABLE INPUT
================================================================================

Purpose:
    Denoises audio from SDR# in real-time. It listens to a virtual audio
    cable (like VB-CABLE), processes the audio through the U-Net model,
    and plays the clean audio to your default speakers.

Setup & Dependencies:
    - VB-CABLE: Install from https://vb-audio.com/Cable/
    - SDR#: Set "Audio" output to "CABLE Input (VB-Audio Virtual Cable)"
    - Python: This script will listen to "CABLE Output"
    - Requires: torch, sounddevice, numpy
    - Model: Needs a trained model (e.g., saved_models/FM/unet1d_best.pth)

Architecture:
    - Audio Thread: Captures virtual cable audio & plays denoised output
    - AI Thread: Processes audio chunks through the U-Net model
    - Queue-based system for smooth, real-time performance

Usage:
    1. Start SDR# and tune to a noisy FM station.
    2. Set SDR# audio output to "CABLE Input".
    3. Run this script:
       python src/live_denoise.py
       → Listens to "CABLE Output"
       → Denoises in real-time
       → Plays clean audio to your default speakers
       → Press Ctrl+C to stop

Troubleshooting:
    - No sound? Check SDR# output and run `python -m sounddevice` to see
      device names. Ensure they match.
    - Stuttering? Increase CHUNK_SIZE or use a CUDA-enabled GPU.
    - Error? Make sure your model is trained and available at the specified path.
================================================================================
"""

class LiveSDRDenoiser:
    """
    Denoises audio from a virtual audio cable in real-time.
    """
    def __init__(self, model_path=None, chunk_size=None, sample_rate=None, device='cpu', 
                 input_device_name='CABLE Output', output_device_name=None, passthrough=False):
        """
        Args:
            input_device_name: Name of the virtual cable output device.
            output_device_name: Name of the speaker/headphone output device (None for default).
            passthrough: If True, bypasses the AI model for testing audio routing.
        """
        # Use config defaults if not specified
        if model_path is None:
            model_path = str(Paths.MODEL_FM_BEST) # Use FM-trained model
        if chunk_size is None:
            chunk_size = AudioSettings.CHUNK_SIZE
        if sample_rate is None:
            sample_rate = AudioSettings.SAMPLE_RATE
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.input_device_name = input_device_name
        self.output_device_name = output_device_name
        self.passthrough = passthrough
        
        # Load model only if not in passthrough mode
        if not self.passthrough:
            print(f"Loading model from {model_path}...")
            self.model = UNet1D(in_channels=1, out_channels=1).to(self.device)
            
            if not Path(model_path).exists():
                print(f"❌ FATAL: Model not found at {model_path}")
                print("Please train the model first using 'python src/fm/model/backshot.py'")
                sys.exit(1)
                
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✅ Model loaded! (Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.6f})")
        else:
            self.model = None
            print("🔊 PASSTHROUGH MODE: AI model is disabled.")

        # Queues
        self.in_queue = Queue(maxsize=20)
        self.out_queue = Queue(maxsize=20)
        
        # Control flags
        self.running = Event()
        self.ai_thread = None
        
        print(f"\nConfiguration:")
        print(f"  Input Device: '{input_device_name}'")
        print(f"  Output Device: '{output_device_name or 'Default'}'")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  Chunk Size: {chunk_size} samples (~{chunk_size/sample_rate*1000:.1f} ms)")
        print(f"  Device: {device.upper()}")
        if self.passthrough:
            print("  Mode: Passthrough (No Denoising)")
        else:
            print("  Mode: Denoising")

    def _find_device_id(self, name, kind='input'):
        """Find audio device ID by its name."""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if name.lower() in device['name'].lower() and device[f'max_{kind}_channels'] > 0:
                print(f"Found {kind} device '{device['name']}' with ID {i}")
                return i
        return None

    def _ai_worker(self):
        """AI Thread: Processes audio chunks."""
        print("🤖 AI thread started")
        
        # Passthrough mode: just move audio from input to output
        if self.passthrough:
            while self.running.is_set():
                if not self.in_queue.empty():
                    chunk = self.in_queue.get()
                    if not self.out_queue.full():
                        self.out_queue.put(chunk)
            print("🤖 AI thread stopped (Passthrough)")
            return

        # Denoising mode
        overlap = self.chunk_size // 2
        buffer = np.zeros(self.chunk_size + overlap, dtype=np.float32)
        
        with torch.no_grad():
            while self.running.is_set():
                if not self.in_queue.empty():
                    noisy_chunk = self.in_queue.get()
                    
                    # Add new chunk to buffer
                    buffer[:-self.chunk_size] = buffer[self.chunk_size:]
                    buffer[-self.chunk_size:] = noisy_chunk
                    
                    # Process the full buffer
                    noisy_tensor = torch.from_numpy(buffer).unsqueeze(0).unsqueeze(0).to(self.device)
                    clean_tensor = self.model(noisy_tensor)
                    clean_buffer = clean_tensor.squeeze().cpu().numpy()
                    
                    # Output only the first part of the processed buffer to avoid overlap artifacts
                    output_chunk = clean_buffer[:self.chunk_size]
                    
                    if not self.out_queue.full():
                        self.out_queue.put(output_chunk)
        
        print("🤖 AI thread stopped (Denoising)")
    
    def _audio_callback(self, indata, outdata, frames, time, status):
        """Audio Thread: Handles audio I/O."""
        if status:
            print(f"⚠️ Audio status: {status}", file=sys.stderr)
        
        # Put incoming audio into the queue
        if not self.in_queue.full():
            self.in_queue.put(indata[:, 0])
        
        # Get denoised audio from the queue
        if not self.out_queue.empty():
            clean_chunk = self.out_queue.get()
            outdata[:, 0] = clean_chunk
        else:
            outdata.fill(0) # Output silence if queue is empty
    
    def start(self):
        """Start the live denoising stream."""
        print("\n" + "="*60)
        print("Starting Live SDR Denoiser")
        print("="*60)
        print("📡 SDR# -> 🎤 CABLE Input -> 🐍 Python -> 🤖 AI Denoising -> 🔊 Speakers")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Find device IDs
        input_device_id = self._find_device_id(self.input_device_name, 'input')
        if input_device_id is None:
            print(f"❌ FATAL: Input device '{self.input_device_name}' not found!")
            print("Please check your VB-CABLE installation and SDR# output.")
            print("Available devices:")
            print(sd.query_devices())
            return
            
        output_device_id = self.output_device_name
        if isinstance(self.output_device_name, str):
            output_device_id = self._find_device_id(self.output_device_name, 'output')

        self.running.set()
        self.ai_thread = Thread(target=self._ai_worker, daemon=True)
        self.ai_thread.start()
        
        try:
            with sd.Stream(
                device=(input_device_id, output_device_id),
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype='float32',
                callback=self._audio_callback
            ):
                print("\n✅ Real-time denoising active!")
                print("   Listening for audio from SDR#...")
                while self.running.is_set():
                    sd.sleep(100)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping...")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the denoiser."""
        if not self.running.is_set():
            return
        print("\nShutting down...")
        self.running.clear()
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)
        print("✅ Stopped successfully!")


def main():
    """Main function to run the live denoiser."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live SDR Audio Denoiser')
    parser.add_argument('--list-devices', action='store_true', help='List all available audio devices and exit.')
    parser.add_argument('--passthrough', action='store_true', help='Bypass AI and pass audio directly through to test routing.')
    parser.add_argument('--input-device', type=str, default='CABLE Output', help='Name of the VB-CABLE input device.')
    parser.add_argument('--output-device', type=str, default=None, help='Name of the output speaker device (optional).')
    parser.add_argument('--model', type=str, default=str(Paths.MODEL_FM_BEST), help='Path to the trained model.')
    parser.add_argument('--chunk-size', type=int, default=8192, help='Audio chunk size.')
    parser.add_argument('--cpu', action='store_true', help='Force CPU processing.')
    args = parser.parse_args()

    if args.list_devices:
        print("="*60)
        print("Available Audio Devices")
        print("="*60)
        print(sd.query_devices())
        print("="*60)
        print("Check the 'name' of your VB-CABLE (e.g., 'CABLE Output') and your speakers.")
        print("Use these names with --input-device and --output-device if needed.")
        return

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Pass passthrough flag to the denoiser
    denoiser = LiveSDRDenoiser(
        model_path=args.model,
        chunk_size=args.chunk_size,
        sample_rate=AudioSettings.SAMPLE_RATE,
        device=device,
        input_device_name=args.input_device,
        output_device_name=args.output_device,
        passthrough=args.passthrough
    )
    
    denoiser.start()


if __name__ == "__main__":
    try:
        import sounddevice as sd
    except ImportError:
        print("❌ sounddevice not installed! Please run: pip install sounddevice")
        sys.exit(1)
    
    main()
