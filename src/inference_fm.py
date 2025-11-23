import torch
import librosa
import soundfile as sf
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import Paths, AudioSettings
from src.fm.model.neuralnet import UNet1D

MODEL_PATH = str(Paths.MODEL_FM_BEST)
DEFAULT_INPUT_DIR = r"Tests\samples\Arti"
DEFAULT_OUTPUT_DIR = r"Tests\tests\testing9-speech"
SAMPLE_RATE = 44100
"""
================================================================================
AUDIO DENOISER - INFERENCE MODULE
================================================================================

Purpose:
    Batch inference script for denoising audio files using trained 1D U-Net model.
    Supports single file or directory batch processing.

Setup & Dependencies:
    - Requires: torch, librosa, soundfile
    - Model checkpoint: saved_models/unet1d_best.pth
    - Audio format: Any format librosa supports (wav, flac, mp3, etc.)
    - Sample rate: 22050 Hz for speech (configured in config.py)

Usage Examples:

    1. SINGLE FILE DENOISING:
       python src/inference.py path/to/noisy_audio.wav
       → Outputs: path/to/denoised_noisy_audio.wav

    2. SINGLE FILE WITH CUSTOM OUTPUT:
       python src/inference.py noisy.wav output_clean.wav
       → Outputs: output_clean.wav

    3. BATCH DENOISING (all .flac files in directory):
       python src/inference.py
       → Reads from: testing/
       → Outputs to: testing/out/denoised_*.flac

How to Use in Your Code:

    from src.inference import Denoiser
    
    # Initialize denoiser
    denoiser = Denoiser(model_path="saved_models/unet1d_best.pth", device='cuda')
    
    # Denoise a file
    denoiser.denoise_file("noisy_audio.wav", "clean_audio.wav", sample_rate=22050)

Model Training:
    - Train with: python src/model/backshot.py
    - Resume from checkpoint: python src/model/backshot.py resume
    - Best model saved to: saved_models/unet1d_best.pth

Created by Satya with Copilot @ 15/11/25

Inference script for 1D U-Net Audio Denoiser
- Loads a trained model
- Denoises audio files
- Saves the cleaned output
================================================================================
"""

class Denoiser:
    """
    Audio denoiser using trained 1D U-Net model.
    """
    def __init__(self, model_path=None, device='cpu'):
        """
        Initialize the denoiser.
        
        Args:
            model_path: Path to the trained model checkpoint (default: uses config)
            device: 'cuda' or 'cpu'
        """
        # Use config default if not specified
        if model_path is None:
            model_path = str(Paths.MODEL_BEST)
        
        self.device = torch.device(device)
        self.model = UNet1D(in_channels=1, out_channels=1).to(self.device)
        self.sample_rate = AudioSettings.SAMPLE_RATE
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
        print(f"   Training loss: {checkpoint['train_loss']:.6f}")
        print(f"   Validation loss: {checkpoint['val_loss']:.6f}")
    
    def denoise_audio(self, audio, sample_rate, chunk_size=16000):
        """
        Denoise an audio signal.
        
        Args:
            audio: Numpy array of audio samples
            sample_rate: Sample rate of the audio
            chunk_size: Size of chunks to process (must match training length)
        
        Returns:
            Denoised audio as numpy array
        """
        # Process audio in chunks
        num_chunks = int(np.ceil(len(audio) / chunk_size))
        denoised_chunks = []
        
        with torch.no_grad():
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(audio))
                
                # Get chunk
                chunk = audio[start:end]
                
                # Pad if necessary
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
                # Convert to tensor
                chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0)  # (1, 1, chunk_size)
                chunk_tensor = chunk_tensor.to(self.device)
                
                # Denoise
                denoised_chunk = self.model(chunk_tensor)
                
                # Convert back to numpy
                denoised_chunk = denoised_chunk.squeeze().cpu().numpy()
                
                # Remove padding if this was the last chunk
                if end - start < chunk_size:
                    denoised_chunk = denoised_chunk[:end - start]
                
                denoised_chunks.append(denoised_chunk)
        
        # Concatenate all chunks
        denoised_audio = np.concatenate(denoised_chunks)
        
        return denoised_audio
    
    def generate_report(self, input_path, output_path, original_audio, denoised_audio, sr):
        """Generate a visual report comparing original and denoised audio."""
        print("\n📊 Generating Performance Report...")
        
        # Calculate metrics
        noise_removed = original_audio - denoised_audio
        orig_rms = np.sqrt(np.mean(original_audio**2))
        denoised_rms = np.sqrt(np.mean(denoised_audio**2))
        noise_rms = np.sqrt(np.mean(noise_removed**2))
        
        # SNR Estimation (Approximate)
        # Assuming noise is the difference between original and denoised
        snr_improvement = 20 * np.log10(denoised_rms / (noise_rms + 1e-9))
        
        print(f"   Original RMS: {orig_rms:.4f}")
        print(f"   Denoised RMS: {denoised_rms:.4f}")
        print(f"   Noise RMS (Est): {noise_rms:.4f}")
        print(f"   SNR Improvement (Est): {snr_improvement:.2f} dB")
        
        # Plotting
        plt.figure(figsize=(12, 8))
        
        # 1. Waveforms
        plt.subplot(3, 1, 1)
        plt.title(f"Waveform Comparison: {Path(input_path).name}")
        plt.plot(original_audio, label='Original (Noisy)', alpha=0.7, color='orange')
        plt.plot(denoised_audio, label='Denoised (Clean)', alpha=0.8, color='blue')
        plt.legend()
        plt.ylabel("Amplitude")
        
        # 2. Spectrograms
        plt.subplot(3, 2, 3)
        plt.title("Original Spectrogram")
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
        librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(3, 2, 4)
        plt.title("Denoised Spectrogram")
        D_denoised = librosa.amplitude_to_db(np.abs(librosa.stft(denoised_audio)), ref=np.max)
        librosa.display.specshow(D_denoised, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        
        # 3. Noise Profile
        plt.subplot(3, 1, 3)
        plt.title("Estimated Noise Profile (Original - Denoised)")
        plt.plot(noise_removed, color='red', alpha=0.6)
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        
        plt.tight_layout()
        
        # Save plot
        report_path = str(Path(output_path).parent / f"report_{Path(output_path).stem}.png")
        plt.savefig(report_path)
        print(f"   📈 Report saved to: {report_path}")
        plt.close()

    def denoise_file(self, input_path, output_path, sample_rate=22050):
        """
        Denoise an audio file and save the result.
        
        Args:
            input_path: Path to noisy audio file
            output_path: Path to save denoised audio
            sample_rate: Sample rate to use (default: 22050)
        """
        print(f"\nDenoising: {input_path}")
        
        # Load audio
        audio, sr = librosa.load(input_path, sr=sample_rate)
        print(f"  Loaded audio: {len(audio)} samples at {sr} Hz")
        
        # Denoise
        denoised = self.denoise_audio(audio, sr)
        print(f"  Denoised: {len(denoised)} samples")
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from output file extension
        output_ext = Path(output_path).suffix.lower()
        if output_ext == '.flac':
            sf.write(output_path, denoised, sr, format='FLAC')
        else:
            # Default to WAV for .wav or unknown extensions
            sf.write(output_path, denoised, sr, format='WAV')
        print(f"  ✅ Saved to: {output_path}")
        
        # Generate Report
        self.generate_report(input_path, output_path, audio, denoised, sr)


def main():
    """
    Example usage: Denoise all files in a directory or a single file.
    
    Usage:
        python src/inference.py                              # Denoise all files in default directory
        python src/inference.py path/to/audio.flac           # Denoise a single file
        python src/inference.py path/to/audio.flac output.flac  # Denoise with custom output path
    """
    # Configuration

    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please train the model first using backshot.py")
        return
    
    # Create denoiser
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    denoiser = Denoiser(MODEL_PATH, device=device)
    
    # Check if user provided a specific file path
    if len(sys.argv) > 1:
        # Single file mode
        input_file = sys.argv[1]
        
        if not Path(input_file).exists():
            print(f"❌ File not found: {input_file}")
            return
        
        # Determine output path
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
        else:
            # Auto-generate output path
            input_path = Path(input_file)
            output_file = str(input_path.parent / f"denoised_{input_path.name}")
        
        print(f"\n{'='*60}")
        print(f"Denoising single file")
        print(f"{'='*60}\n")
        
        denoiser.denoise_file(input_file, output_file, SAMPLE_RATE)
        
        print(f"\n{'='*60}")
        print(f"✅ File denoised!")
        print(f"Output: {output_file}")
        print(f"{'='*60}\n")
    else:
        # Batch mode - denoise all files in directory
        INPUT_DIR = DEFAULT_INPUT_DIR
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
        
        # Get all noisy files (.wav and .flac)
        input_files = list(Path(INPUT_DIR).glob("*.flac")) + list(Path(INPUT_DIR).glob("*.wav"))
        
        if not input_files:
            print(f"❌ No .flac or .wav files found in {INPUT_DIR}")
            return
        
        print(f"\n{'='*60}")
        print(f"Denoising {len(input_files)} files")
        print(f"{'='*60}\n")
        
        # Denoise each file
        for i, input_path in enumerate(input_files, 1):
            output_path = Path(OUTPUT_DIR) / f"denoised_{input_path.name}"
            print(f"[{i}/{len(input_files)}]")
            denoiser.denoise_file(str(input_path), str(output_path), SAMPLE_RATE)
        
        print(f"\n{'='*60}")
        print(f"✅ All files denoised!")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
