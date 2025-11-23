import torch
import numpy as np
import sounddevice as sd
from queue import Queue
from threading import Thread, Event
import sys
from pathlib import Path
import soundfile as sf
import time
import matplotlib.pyplot as plt
import librosa
import librosa.display
from collections import deque
from datetime import datetime

def select_fm_model():
    """Interactive model selection for FM denoising."""
    model_dir = Path('saved_models/FM/FM_Final_1DUNET')
    
    if not model_dir.exists():
        print(f"⚠️  Model directory not found: {model_dir}")
        default_model = str(Paths.MODEL_FM_BEST)
        print(f"   Using default: {default_model}")
        return default_model
    
    # Find all .pth files
    models = sorted(model_dir.glob('*.pth'))
    
    if not models:
        print(f"⚠️  No models found in {model_dir}")
        default_model = str(Paths.MODEL_FM_BEST)
        print(f"   Using default: {default_model}")
        return default_model
    
    print("\n" + "="*60)
    print("🧠 SELECT FM DENOISING MODEL")
    print("="*60)
    print("\nAvailable models:")
    
    for i, model_path in enumerate(models, 1):
        model_name = model_path.stem
        if 'general' in model_name.lower():
            desc = "(All-purpose denoising)"
        elif 'music' in model_name.lower():
            desc = "(Optimized for music/FM radio)"
        elif 'speech' in model_name.lower():
            desc = "(Optimized for voice/talk radio)"
        else:
            desc = ""
        print(f"  [{i}] {model_path.name} {desc}")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (1-{len(models)}): ")
            choice = int(choice)
            
            if 1 <= choice <= len(models):
                selected = str(models[choice - 1])
                print(f"✅ Selected: {selected}")
                return selected
            else:
                print("❌ Invalid choice. Try again.")
        except (ValueError, KeyboardInterrupt):
            print("\n⚠️  Using default model (music)")
            return str(Paths.MODEL_FM_BEST)

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

        # Reporting buffers (Keep last 10 seconds)
        self.history_duration = 10 
        # Calculate max chunks: (Duration * SR) / ChunkSize
        history_len = int((self.history_duration * self.sample_rate) / self.chunk_size)
        self.input_history = deque(maxlen=history_len)
        self.output_history = deque(maxlen=history_len)

    def generate_report(self):
        """Generate a performance report from the captured history."""
        if not self.input_history or not self.output_history:
            print("\n⚠️ Not enough data to generate report.")
            return

        print("\n📊 Generating Live Session Report...")
        
        # Convert history to arrays
        # Deque contains chunks, so we need to concatenate
        original = np.concatenate(list(self.input_history))
        denoised = np.concatenate(list(self.output_history))
        
        # Ensure lengths match (they should, but just in case)
        min_len = min(len(original), len(denoised))
        original = original[:min_len]
        denoised = denoised[:min_len]
        
        # Calculate metrics
        noise_removed = original - denoised
        orig_rms = np.sqrt(np.mean(original**2))
        denoised_rms = np.sqrt(np.mean(denoised**2))
        noise_rms = np.sqrt(np.mean(noise_removed**2))
        
        # SNR Estimation
        snr_improvement = 20 * np.log10(denoised_rms / (noise_rms + 1e-9))
        
        print(f"   Captured Duration: {len(original)/self.sample_rate:.2f}s")
        print(f"   Original RMS: {orig_rms:.4f}")
        print(f"   Denoised RMS: {denoised_rms:.4f}")
        print(f"   SNR Improvement (Est): {snr_improvement:.2f} dB")
        
        # Plotting
        try:
            plt.figure(figsize=(12, 8))
            
            # 1. Waveforms
            plt.subplot(3, 1, 1)
            plt.title("Waveform Comparison (Last 10s)")
            plt.plot(original, label='Original (Noisy)', alpha=0.7, color='orange')
            plt.plot(denoised, label='Denoised (Clean)', alpha=0.8, color='blue')
            plt.legend()
            plt.ylabel("Amplitude")
            
            # 2. Spectrograms
            plt.subplot(3, 2, 3)
            plt.title("Original Spectrogram")
            D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original)), ref=np.max)
            librosa.display.specshow(D_orig, sr=self.sample_rate, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            
            plt.subplot(3, 2, 4)
            plt.title("Denoised Spectrogram")
            D_denoised = librosa.amplitude_to_db(np.abs(librosa.stft(denoised)), ref=np.max)
            librosa.display.specshow(D_denoised, sr=self.sample_rate, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            
            # 3. Noise Profile
            plt.subplot(3, 1, 3)
            plt.title("Estimated Noise Profile")
            plt.plot(noise_removed, color='red', alpha=0.6)
            plt.xlabel("Time (samples)")
            plt.ylabel("Amplitude")
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"live_report_{timestamp}.png"
            plt.savefig(report_path)
            print(f"   📈 Report saved to: {report_path}")
            plt.close()
        except Exception as e:
            print(f"   ❌ Failed to generate plot: {e}")

    def _find_device_id(self, name, kind='input'):
        """Find audio device ID by its name."""
        devices = sd.query_devices()
        print(f"\n🔍 Searching for {kind} device matching '{name}'...")
        for i, device in enumerate(devices):
            if name.lower() in device['name'].lower() and device[f'max_{kind}_channels'] > 0:
                print(f"   ✅ Found: ID {i} -> '{device['name']}'")
                return i
        print(f"   ❌ NOT FOUND! Available {kind} devices:")
        for i, device in enumerate(devices):
            if device[f'max_{kind}_channels'] > 0:
                print(f"      ID {i}: {device['name']}")
        return None

    def _ai_worker(self):
        """AI Thread: Processes audio chunks."""
        print("\n🤖 AI thread started")
        processed_count = 0
        
        # Passthrough mode: just move audio from input to output
        if self.passthrough:
            print("✅ PASSTHROUGH MODE ACTIVE - Audio will pass through without AI processing")
            while self.running.is_set():
                if not self.in_queue.empty():
                    chunk = self.in_queue.get()
                    processed_count += 1
                    
                    # Record history
                    self.input_history.append(chunk)
                    self.output_history.append(chunk)
                    
                    if not self.out_queue.full():
                        self.out_queue.put(chunk)
                    else:
                        print(f"⚠️ Output queue full in passthrough mode! Chunk #{processed_count} dropped.")
            print(f"\n🤖 AI thread stopped (Passthrough). Processed {processed_count} chunks.")
            return

        # Denoising mode
        print(f"✅ DENOISING MODE ACTIVE - Audio will be processed through AI model")
        overlap = self.chunk_size // 2
        buffer = np.zeros(self.chunk_size + overlap, dtype=np.float32)
        processed_count = 0
        
        with torch.no_grad():
            while self.running.is_set():
                if not self.in_queue.empty():
                    noisy_chunk = self.in_queue.get()
                    processed_count += 1
                    
                    # Add new chunk to buffer
                    buffer[:-self.chunk_size] = buffer[self.chunk_size:]
                    buffer[-self.chunk_size:] = noisy_chunk
                    
                    # Process the full buffer
                    noisy_tensor = torch.from_numpy(buffer).unsqueeze(0).unsqueeze(0).to(self.device)
                    clean_tensor = self.model(noisy_tensor)
                    clean_buffer = clean_tensor.squeeze().cpu().numpy()
                    
                    # Output only the first part of the processed buffer to avoid overlap artifacts
                    output_chunk = clean_buffer[:self.chunk_size]
                    
                    # Record history
                    self.input_history.append(noisy_chunk)
                    self.output_history.append(output_chunk)
                    
                    if not self.out_queue.full():
                        self.out_queue.put(output_chunk)
                    else:
                        print(f"⚠️ Output queue full! Denoised chunk #{processed_count} dropped.")
        
        print(f"\n🤖 AI thread stopped (Denoising). Processed {processed_count} chunks.")
    
    def _audio_callback(self, indata, outdata, frames, time, status):
        """Audio Thread: Handles audio I/O."""
        if status:
            print(f"⚠️ Audio status: {status}", file=sys.stderr)
        
        # Get audio volume for debug
        input_volume = np.abs(indata[:, 0]).max()
        
        # Put incoming audio into the queue
        if not self.in_queue.full():
            self.in_queue.put(indata[:, 0].copy())
        else:
            print(f"⚠️ Input queue full! Dropping audio chunk. In: {self.in_queue.qsize()}, Out: {self.out_queue.qsize()}")
        
        # Get processed audio from the queue
        if not self.out_queue.empty():
            clean_chunk = self.out_queue.get()
            outdata[:, 0] = clean_chunk
            output_volume = np.abs(clean_chunk).max()
            print(f"🔊 IN: {input_volume:.4f} | OUT: {output_volume:.4f} | Q(in): {self.in_queue.qsize()} | Q(out): {self.out_queue.qsize()}")
        else:
            outdata.fill(0)  # Output silence if queue is empty
            if input_volume > 0:
                print(f"⏳ Waiting for processing... IN: {input_volume:.4f} | Q(in): {self.in_queue.qsize()} | Q(out): {self.out_queue.qsize()}")
    
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
            
        output_device_id = None
        if isinstance(self.output_device_name, str):
            output_device_id = self._find_device_id(self.output_device_name, 'output')
        elif self.output_device_name is not None:
            output_device_id = self.output_device_name

        self.running.set()
        self.ai_thread = Thread(target=self._ai_worker, daemon=True)
        self.ai_thread.start()
        
        # --- Live Plot Setup (DISABLED) ---
        # print("📊 Initializing Live Plot...")
        # plt.ion()
        # fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(10, 8))
        # fig.canvas.manager.set_window_title("Live SDR Audio Monitor")
        # 
        # # Waveform
        # x = np.arange(self.chunk_size)
        # line_noisy, = ax_wave.plot(x, np.zeros(self.chunk_size), color='orange', alpha=0.6, label='Noisy Input')
        # line_clean, = ax_wave.plot(x, np.zeros(self.chunk_size), color='blue', alpha=0.8, label='Clean Output')
        # ax_wave.set_ylim(-0.5, 0.5) 
        # ax_wave.set_title("Real-time Waveform")
        # ax_wave.legend(loc='upper right')
        # ax_wave.grid(True, alpha=0.3)
        # 
        # # Spectrum
        # ax_spec.set_title("Real-time Frequency Spectrum")
        # ax_spec.set_xlabel("Frequency (Hz)")
        # ax_spec.set_ylabel("Magnitude (dB)")
        # ax_spec.set_xlim(0, self.sample_rate // 2)
        # ax_spec.set_ylim(-100, 0)
        # ax_spec.grid(True, alpha=0.3)
        # 
        # freqs = np.fft.rfftfreq(self.chunk_size, 1/self.sample_rate)
        # line_spec_noisy, = ax_spec.plot(freqs, np.zeros_like(freqs), color='orange', alpha=0.6, label='Noisy')
        # line_spec_clean, = ax_spec.plot(freqs, np.zeros_like(freqs), color='blue', alpha=0.8, label='Clean')
        # ax_spec.legend(loc='upper right')
        # 
        # plt.tight_layout()
        # -----------------------
        
        # Get device info to check sample rate compatibility
        devices = sd.query_devices()
        input_info = devices[input_device_id]
        output_info = devices[output_device_id] if output_device_id else devices[sd.default.device[1]]
        
        print(f"\n🔧 Device Configuration:")
        print(f"   Input:  {input_info['name']}")
        print(f"          Default SR: {input_info['default_samplerate']} Hz")
        print(f"   Output: {output_info['name']}")
        print(f"          Default SR: {output_info['default_samplerate']} Hz")
        print(f"   Using:  {self.sample_rate} Hz\n")
        
        try:
            with sd.Stream(
                device=(input_device_id, output_device_id),
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype='float32',
                callback=self._audio_callback,
                prime_output_buffers_using_stream_callback=True
            ):
                print(f"\n✅ Audio stream active!")
                print(f"   Input:  Device #{input_device_id} ('{self.input_device_name}')")
                print(f"   Output: Device #{output_device_id if output_device_id else 'Default'}")
                print(f"   Mode:   {'PASSTHROUGH (No AI)' if self.passthrough else 'DENOISING (AI Active)'}")
                print(f"   Listening for audio... Press Ctrl+C to stop.\n")
                
                while self.running.is_set():
                    time.sleep(0.1)  # Just wait, no plot updates
                    # Live plot disabled - audio still processing in background
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping...")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
        finally:
            # plt.close('all')  # Disabled since no plot
            self.stop()
    
    def stop(self):
        """Stop the denoiser."""
        if not self.running.is_set():
            return
        print("\nShutting down...")
        self.running.clear()
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)
        
        # Generate report on exit
        self.generate_report()
        
        print("✅ Stopped successfully!")


def main():
    """Main function to run the live denoiser."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live SDR Audio Denoiser')
    parser.add_argument('--list-devices', action='store_true', help='List all available audio devices and exit.')
    parser.add_argument('--passthrough', action='store_true', help='Bypass AI and pass audio directly through to test routing.')
    parser.add_argument('--input-device', type=str, default='CABLE Output', help='Name of the VB-CABLE input device.')
    parser.add_argument('--output-device', type=str, default=None, help='Name of the output speaker device. Set to your preferred speaker name or leave None for system default.')
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

    # Interactive model selection (unless passthrough mode)
    if not args.passthrough:
        selected_model = select_fm_model()
        args.model = selected_model

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
