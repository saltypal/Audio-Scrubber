import torch
import librosa
import soundfile as sf
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_music import Paths, AudioSettings
from src.fm.model.neuralnet import UNet1D

MODEL_PATH = str(Paths.MODEL_MUSIC_BEST)
DEFAULT_INPUT_DIR = str(Paths.MUSIC_ROOT)
DEFAULT_OUTPUT_DIR = str(Paths.MUSIC_DENOISED_OUTPUT)
SAMPLE_RATE = AudioSettings.SAMPLE_RATE


class Denoiser:
    """Music audio denoiser using trained 1D U-Net model."""

    def __init__(self, model_path=None, device='cpu'):
        if model_path is None:
            model_path = MODEL_PATH

        self.device = torch.device(device)
        self.model = UNet1D(in_channels=1, out_channels=1).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.sample_rate = SAMPLE_RATE

        print(f"[OK] Music model loaded from {model_path}")
        print(f"   Training loss: {checkpoint.get('train_loss', 0.0):.6f}")
        print(f"   Validation loss: {checkpoint.get('val_loss', 0.0):.6f}")

    def denoise_audio(self, audio, sample_rate, chunk_size=AudioSettings.AUDIO_LENGTH):
        num_chunks = int(np.ceil(len(audio) / chunk_size))
        denoised_chunks = []

        with torch.no_grad():
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(audio))

                chunk = audio[start:end]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

                chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0)
                chunk_tensor = chunk_tensor.to(self.device)

                denoised_chunk = self.model(chunk_tensor)
                denoised_chunk = denoised_chunk.squeeze().cpu().numpy()

                if end - start < chunk_size:
                    denoised_chunk = denoised_chunk[:end - start]

                denoised_chunks.append(denoised_chunk)

        return np.concatenate(denoised_chunks)

    def denoise_file(self, input_path, output_path, sample_rate=None):
        if sample_rate is None:
            sample_rate = SAMPLE_RATE

        print(f"\nDenoising music file: {input_path}")
        audio, sr = librosa.load(input_path, sr=sample_rate)
        print(f"  Loaded: {len(audio)} samples at {sr} Hz")

        denoised = self.denoise_audio(audio, sr)
        print(f"  Denoised: {len(denoised)} samples")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, denoised, sr, format='FLAC')
        print(f"  [OK] Saved to: {output_path}")


def main():
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Music model not found at {MODEL_PATH}")
        print("Please train the music model first using backshot_music.py")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    denoiser = Denoiser(MODEL_PATH, device=device)

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if not Path(input_file).exists():
            print(f"[ERROR] File not found: {input_file}")
            return

        if len(sys.argv) > 2:
            output_file = sys.argv[2]
        else:
            input_path = Path(input_file)
            output_file = str(input_path.parent / f"denoised_{input_path.name}")

        print("\n============================================================")
        print("Denoising single music file")
        print("============================================================\n")
        denoiser.denoise_file(input_file, output_file)

        print("\n============================================================")
        print("[SUCCESS] File denoised!")
        print(f"Output: {output_file}")
        print("============================================================\n")
    else:
        input_dir = Path(DEFAULT_INPUT_DIR)
        output_dir = Path(DEFAULT_OUTPUT_DIR)
        
        # Collect all audio files recursively
        input_files = []
        input_files.extend(list(input_dir.rglob("*.flac")))
        input_files.extend(list(input_dir.rglob("*.wav")))
        input_files.extend(list(input_dir.rglob("*.mp3")))

        if not input_files:
            print(f"[ERROR] No .flac, .wav, or .mp3 files found in {input_dir}")
            print(f"Searched recursively in: {input_dir.resolve()}")
            print(f"Directory exists: {input_dir.exists()}")
            return

        print("\n============================================================")
        print(f"Denoising {len(input_files)} music files from {input_dir}")
        print("============================================================\n")

        for i, input_path in enumerate(input_files, 1):
            output_path = output_dir / f"denoised_{input_path.name}"
            print(f"[{i}/{len(input_files)}]")
            denoiser.denoise_file(str(input_path), str(output_path))

        print("\n============================================================")
        print("[SUCCESS] All music files denoised!")
        print(f"Output directory: {output_dir}")
        print("============================================================\n")


if __name__ == "__main__":
    main()
