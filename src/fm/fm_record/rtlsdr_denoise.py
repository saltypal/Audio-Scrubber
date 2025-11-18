"""
================================================================================
RTL-SDR REAL-TIME AI DENOISER (v3 - Production Ready)
================================================================================
Author: Satya with GitHub Copilot
Date: 17 November 2025

Captures FM radio, denoises with AI, plays clean audio in real-time.
"""
import torch
import numpy as np
import sounddevice as sd
from scipy import signal
from queue import Queue, Empty
from threading import Thread, Event
import sys
from pathlib import Path
import argparse
import asyncio

# Add parent directory to path to allow importing from 'model' and 'config'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from src.fm.model.neuralnet import UNet1D
from config import Paths, AudioSettings, RTLSDRSettings
from src.fm.fm_record.rtlsdr_core import RTLSDRCore


class RTLSDRDenoiser:
    """
    Manages the real-time FM radio denoising pipeline.
    """
    def __init__(self, model_path, frequency, sample_rate, chunk_size, device, gain, ppm):
        self.model_path = model_path
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.gain = gain
        self.ppm = ppm
        self.bypass_ai = False # Will be set by main()

        # Queues act as conveyor belts between threads
        self.raw_audio_queue = Queue(maxsize=20) # Demodulated audio from SDR
        self.processed_audio_queue = Queue(maxsize=20) # Denoised audio for playback

        # Thread control
        self.running = Event()
        self.sdr = None

        self._print_configuration()
        self.model = self._load_model()

    def _print_configuration(self):
        print("\n" + "="*60)
        print("RTL-SDR AI Denoiser Configuration")
        print(f"{'='*60}")
        print(f"  Model Path:      {self.model_path}")
        print(f"  FM Frequency:    {self.frequency / 1e6:.2f} MHz")
        print(f"  Audio Sample Rate: {self.sample_rate / 1e3:.1f} kHz")
        print(f"  Audio Chunk Size:  {self.chunk_size} samples")
        print(f"  Processing Device: {self.device.type.upper()}")
        print(f"  SDR Gain:        {self.gain}")
        print(f"  SDR PPM Offset:  {self.ppm}")
        print(f"{'='*60}\n")

    def _load_model(self):
        """Loads the U-Net model from the specified path."""
        print(f"Loading model from '{self.model_path}'...")
        if not Path(self.model_path).exists():
            print(f"[ERROR] Model file not found at '{self.model_path}'")
            sys.exit(1)
        try:
            model = UNet1D(in_channels=1, out_channels=1).to(self.device)
            # Use weights_only=True for security
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            
            # Handle both checkpoint dict and raw state_dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                val_loss = checkpoint.get('val_loss', -1)
                print(f"[SUCCESS] Model loaded successfully! (Validation Loss: {val_loss:.6f})")
            else:
                model.load_state_dict(checkpoint)
                print("[SUCCESS] Model loaded successfully! (Direct state_dict)")

            model.eval()
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load model. Error: {e}")
            sys.exit(1)

    def start(self):
        """Starts the SDR capture, AI processing, and audio playback threads."""
        print("Starting all threads...")
        self.running.set()

        # Define threads
        sdr_thread = Thread(target=self._sdr_worker, daemon=True)
        ai_thread = Thread(target=self._ai_worker, daemon=True)
        audio_thread = Thread(target=self._audio_worker, daemon=True)

        # Start threads
        sdr_thread.start()
        ai_thread.start()
        audio_thread.start()

        print("[INFO] All threads running. Listening for FM radio...")
        print("   Press Ctrl+C to stop.")

        # Keep main thread alive to catch KeyboardInterrupt
        try:
            while self.running.is_set():
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\n\n[INFO] Ctrl+C detected. Shutting down...")
        finally:
            self.stop()

    def stop(self):
        """Stops all running threads gracefully."""
        self.running.clear()
        # Add sentinel values to unblock queues
        self.raw_audio_queue.put(None)
        self.processed_audio_queue.put(None)
        print("All threads stopped.")

    def _sdr_worker(self):
        """Thread: Capture FM radio with async RTL-SDR."""
        print("[SDR] Starting SDR thread...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._sdr_async())

    async def _sdr_async(self):
        """Async SDR capture using proper async streaming."""
        try:
            from rtlsdr.rtlsdraio import RtlSdrAio
            
            # Use async version of RtlSdr
            self.sdr = RtlSdrAio()
            await self.sdr.open()
            
            self.sdr.sample_rate = 250000
            self.sdr.center_freq = self.frequency
            self.sdr.gain = self.gain if self.gain != 'auto' else 'auto'

            decimation_rate = int(self.sdr.sample_rate / self.sample_rate)
            
            print(f"[SUCCESS] SDR Connected")
            print(f"   Tuned to {self.sdr.center_freq / 1e6:.2f} MHz")
            print(f"   Sample Rate: {self.sdr.sample_rate / 1e6:.2f} MHz")
            print(f"[INFO] Starting audio stream capture...")

            # Use async streaming (proper method)
            chunk_size = 240000
            
            async for samples in self.sdr.stream(chunk_size):
                if not self.running.is_set():
                    break

                # FM Demodulation
                angle_diff = np.angle(samples[1:] * np.conj(samples[:-1]))

                # Decimation
                audio_raw = signal.decimate(angle_diff, decimation_rate)

                # Normalization
                audio_chunk = np.float32(audio_raw[:self.chunk_size])
                audio_chunk /= np.pi

                if len(audio_chunk) < self.chunk_size:
                    padding = np.zeros(self.chunk_size - len(audio_chunk), dtype=np.float32)
                    audio_chunk = np.concatenate((audio_chunk, padding))

                try:
                    self.raw_audio_queue.put(audio_chunk, timeout=0.1)
                except:
                    pass  # Drop frames if queue is full
            
            await self.sdr.stop()
            self.sdr.close()

        except Exception as e:
            print(f"[ERROR] SDR Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[SDR] SDR thread stopped.")
            self.running.clear()

    def _ai_worker(self):
        """
        Thread 2: Takes raw audio from raw_audio_queue, runs it through the
        AI model, and puts the clean audio into the processed_audio_queue.
        This version buffers audio to match the model's training length.
        """
        print("[AI] AI processing thread started.")
        
        if self.bypass_ai:
            print("   -- AI Bypassed --")
            while self.running.is_set():
                raw_chunk = self.raw_audio_queue.get()
                if raw_chunk is None: break # Sentinel
                self.processed_audio_queue.put(raw_chunk)
            print("[AI] AI processing thread stopped.")
            return

        input_buffer = np.array([], dtype=np.float32)
        model_frame_size = AudioSettings.AUDIO_LENGTH # e.g., 44096

        with torch.no_grad():
            while self.running.is_set():
                # Get a chunk of raw audio from the SDR thread
                raw_chunk = self.raw_audio_queue.get()
                if raw_chunk is None: # Sentinel to stop the thread
                    break

                # Add new chunk to our buffer
                input_buffer = np.concatenate((input_buffer, raw_chunk))

                # If we have enough data for a full model frame, process it
                while len(input_buffer) >= model_frame_size:
                    # Get the frame to process
                    frame_to_process = input_buffer[:model_frame_size]
                    
                    # Remove the processed frame from the start of the buffer
                    input_buffer = input_buffer[model_frame_size:]

                    # Reshape for the model: [batch, channels, length]
                    noisy_tensor = torch.from_numpy(frame_to_process).unsqueeze(0).unsqueeze(0).to(self.device)

                    # Denoise with the U-Net model
                    clean_tensor = self.model(noisy_tensor)

                    # Reshape back to a simple audio chunk
                    clean_chunk = clean_tensor.squeeze().cpu().numpy()

                    # Put the clean audio into the queue for playback
                    self.processed_audio_queue.put(clean_chunk)

        print("[AI] AI processing thread stopped.")

    def _audio_worker(self):
        """
        Thread 3: Takes clean audio from the processed_audio_queue and plays it
        to the default speakers. This version handles variable-sized chunks.
        """
        print("[AUDIO] Audio playback thread started.")
        try:
            def audio_callback(outdata, frames, time, status):
                if status:
                    print(f"[WARNING] Audio playback status: {status}", file=sys.stderr)
                try:
                    # Get a chunk of clean audio from the AI thread
                    processed_chunk = self.processed_audio_queue.get_nowait()
                    if processed_chunk is None: # Sentinel
                        raise sd.CallbackStop
                    
                    chunk_len = len(processed_chunk)
                    outdata[:chunk_len, 0] = processed_chunk
                    if chunk_len < frames:
                        outdata[chunk_len:, 0] = 0 # Fill rest with silence

                except Empty:
                    # If the queue is empty, play silence to avoid stuttering
                    outdata.fill(0)

            # Start the audio stream with a larger blocksize to handle the buffered output
            with sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=AudioSettings.AUDIO_LENGTH, # Match the model output size
                channels=1,
                dtype='float32',
                callback=audio_callback
            ):
                while self.running.is_set():
                    sd.sleep(100) # Keep the stream alive

        except Exception as e:
            print(f"[ERROR] Audio Playback Error: {e}", file=sys.stderr)
        finally:
            print("[AUDIO] Audio playback thread stopped.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="RTL-SDR Real-Time AI Denoiser")
    parser.add_argument('--model', type=str, default=str(Paths.MODEL_BEST),
                        help='Path to the trained model checkpoint.')
    parser.add_argument('--frequency', type=float, default=RTLSDRSettings.FM_FREQUENCY,
                        help='FM frequency to tune to in Hz (e.g., 88.5e6).')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Processing device: 'cuda' or 'cpu'.")
    parser.add_argument('--gain', default='auto',
                        help="SDR gain. 'auto' or a value in dB (e.g., 40.2).")
    parser.add_argument('--ppm', type=int, default=0,
                        help='Frequency correction for the SDR in parts-per-million.')
    parser.add_argument('--no-ai', action='store_true',
                        help='Bypass the AI model to hear the raw, noisy FM signal.')
    args = parser.parse_args()

    # Create and start the denoiser
    denoiser = RTLSDRDenoiser(
        model_path=args.model,
        frequency=args.frequency,
        sample_rate=AudioSettings.SAMPLE_RATE,
        chunk_size=AudioSettings.CHUNK_SIZE,
        device=args.device,
        gain=args.gain,
        ppm=args.ppm
    )
    if args.no_ai:
        denoiser.bypass_ai = True
        
    denoiser.start()


if __name__ == "__main__":
    main()
