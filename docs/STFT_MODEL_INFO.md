================================================================================
STFT-BASED CONV2D MODEL - SUMMARY
================================================================================

Created: November 16, 2025

Files Created:
1. src/model2/stft_net.py - STFT-based 2D U-Net architecture
2. src/model2/backshot_stft.py - Training script for STFT model
3. src/model2/__init__.py - Package initialization

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

Input: STFT Magnitude Spectrograms
- Size: (batch, 1, freq_bins, time_frames)
- Freq bins: 1025 (from n_fft=2048)
- Time frames: Variable (depends on audio duration)

Network: 2D U-Net with Skip Connections
- Encoder: 4 downsampling layers (Conv2D + MaxPool2D)
- Bottleneck: Central feature compression layer
- Decoder: 4 upsampling layers (ConvTranspose2D + skip connections)
- Output: Same shape as input (denoised spectrogram)

Total Parameters: 960,049

Key Features:
✅ Handles variable-sized spectrograms (adaptive padding)
✅ 2D convolutions for frequency-time feature learning
✅ Skip connections preserve fine spectral details
✅ Works in frequency domain (better for stationary noise)

================================================================================
STFT PARAMETERS
================================================================================

FFT Window Size (n_fft): 2048
Hop Length: 512 samples (~23ms at 22050 Hz)
Window Length: 2048
Sample Rate: 22050 Hz
Audio Length: 44096 samples (~2 seconds)

Spectrogram Output:
- Frequency bins: 1025 (2048/2 + 1)
- Time frames: ~83-87 (varies by audio length)

================================================================================
TRAINING CONFIGURATION
================================================================================

Dataset: dataset/instant/
- Clean files: 100
- Noisy files: 500 (5x augmentation)
- Training split: 80/20
- Training samples: 400
- Validation samples: 100

Training Settings:
- Epochs: 40
- Batch size: 16
- Learning rate: 0.0001
- Optimizer: AdamW with weight decay (1e-5)
- Loss: MSE (Mean Squared Error)
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Early stopping: 6 epochs without improvement

Model Checkpoints:
- Best model: saved_models/stft_unet2d_best.pth
- Latest checkpoint: saved_models/checkpoints/stft_latest_checkpoint.pth
- Training history plot: output/stft_training_history.png

================================================================================
USAGE
================================================================================

Start Training:
    python src/model2/backshot_stft.py

Resume from Checkpoint:
    python src/model2/backshot_stft.py resume

With Specific Device:
    python src/model2/backshot_stft.py --device cuda
    python src/model2/backshot_stft.py --device cpu

================================================================================
ADVANTAGES OVER 1D TIME-DOMAIN MODEL
================================================================================

✅ Better spectral learning: 2D convolutions process frequency-time patterns
✅ Efficient noise removal: Excellent for stationary noise (hum, hiss)
✅ Phase preservation: Can reconstruct phase separately for perfect reconstruction
✅ Interpretable: Each filter learns specific frequency patterns
✅ Reduced computation: Spectrograms are more compact than raw waveforms

Disadvantages:
❌ Requires phase reconstruction for audio output
❌ Slightly more complex to implement inference
❌ STFT inverse operation needed

================================================================================
NEXT STEPS
================================================================================

1. Monitor training progress
   - Check loss curves in output/stft_training_history.png
   - Validate loss converging

2. Create inference script for STFT model
   - Load trained checkpoint
   - Apply STFT to input audio
   - Denoise spectrogram
   - Apply inverse STFT with phase

3. Compare with 1D model
   - Benchmark quality metrics
   - Compare inference speed
   - Analyze noise reduction characteristics

4. Fine-tune hyperparameters if needed
   - Adjust batch size
   - Modify learning rate
   - Experiment with STFT parameters

================================================================================
Dataset Notes
================================================================================

The instant dataset contains smaller audio samples from LibriSpeech for quick
training and testing. The noisy versions have noise added at multiple levels
(amplitude-based naming: noise_level_1-5).

Audio loading uses scipy.signal.spectrogram with fallback to librosa.

================================================================================
