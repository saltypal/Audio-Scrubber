# AudioScrubber — FM Denoising Readme

## Overview
AudioScrubber denoises FM audio in real time and offline using deep-learning U-Nets. It targets noisy SDR captures (RTL-SDR/SDR#) and produces cleaner audio for speech and music. The project centers on FM noise captured from the spectrum and models trained to remove it.

## Hardware Receiver
- Primary receiver: RTL-SDR (USB dongle). SDR# is used to tune FM stations and output baseband audio via VB-CABLE.
- Virtual audio cable: VB-CABLE links SDR# output to our Python pipeline for live denoising.
- Optional front-ends: Airspy or other SDRs can be substituted if they emit PCM audio into VB-CABLE at 44.1 kHz.

## Dataset and Noise
- Clean speech: LibriSpeech dev-clean/dev-other at 44.1 kHz (resampled) as configured in [config.py](config.py#L9-L145).
- Music set: curated music clips stored under `dataset/music` (mono, 44.1 kHz).
- Noise capture: real FM noise recorded by scanning across the FM band; saved as `dataset/noise/FM_Noise.wav`. This is not synthetic white noise but spectrum-sourced interference.
- On-the-fly mixing: during training, clean clips are randomly cropped/padded to 88,192 samples (~2 s) and mixed with FM noise at random SNRs inside the dataloader ([src/fm/1dunet/backshot.py](src/fm/1dunet/backshot.py#L20-L211)). This keeps memory low and diversifies examples.
- Dataset categories / modes:
	- Speech (LibriSpeech speech at 44.1 kHz, FLAC/wav after resample)
	- Music (music clips at 44.1 kHz, FLAC/wav)
	- General (a combined model; shares the same sample rate but broader content)

## Chunking Strategy
- Training segment length: 88,192 samples (~2 s at 44.1 kHz), divisible by 16 for 4 U-Net downsamplings. Longer segments help the model learn broader temporal context and reduce boundary artifacts.
- Live/streaming chunk: typically 8,192 samples (~186 ms) in [src/live_denoise.py](src/live_denoise.py#L1-L220) with fallback to 4,096 in [config.py](config.py#L100-L173). The chunk is large enough to preserve context for denoising yet small enough for low latency. Fallback reuse of the last chunk avoids audible gaps if the AI thread lags.

## Models
- 1D U-Net (waveform): [src/fm/neuralnets/neuralnet.py](src/fm/neuralnets/neuralnet.py). Encoder-decoder with skip connections on raw waveforms.
- STFT 2D U-Net (spectrogram): [src/fm/neuralnets/stft_unet2d.py](src/fm/neuralnets/stft_unet2d.py). Operates on magnitude spectrograms; alternative path, not fully benchmarked in practice.
- Model loader for dynamic selection and architecture detection: [src/fm/model_loader.py](src/fm/model_loader.py#L1-L150).
- Exported weights: `saved_models/FM/FinalModels/FM_Final_1DUNET/{general,music,speech}.pth` and `FM_Final_STFT/{general,music,speech}.pth`.

## Training Pipeline (1D U-Net)
- Script: [src/fm/1dunet/backshot.py](src/fm/1dunet/backshot.py).
- Data: LibriSpeech clean audio; noise: FM_Noise.wav; random SNR per sample.
- Hyperparams: LR 1e-4 Adam, batch 2–4, epochs 50–100, gradient clip 1.0, ReduceLROnPlateau scheduler, checkpoints every 5 epochs, best-model saving on val loss.
- Memory optimizations: lazy noise load, streaming file reads, small batch size, CUDA auto-detect.

## Offline Inference
- Script: [src/inference_fm.py](src/inference_fm.py).
- Usage: point to a noisy file or a folder; outputs denoised audio and a PNG report (waveforms, spectrograms, estimated SNR gain).
- Defaults: model path from config (`saved_models/FM/unet1d_music_best.pth` or final FM models), sample rate 44.1 kHz, chunked processing with padding to the training length.

## Real-Time Pipeline (Detailed)
- Entry: [src/live_denoise.py](src/live_denoise.py).
- Signal chain: RTL-SDR → SDR# (tuned FM) → VB-CABLE “CABLE Input” → Python (sounddevice) captures “CABLE Output” → AI denoiser → speakers.
- Threads/queues: audio callback enqueues chunks; AI thread dequeues, denoises with selected model, re-enqueues for playback. Last-output fallback prevents gaps. History buffers feed post-run reports.
- Model selection: auto-discovers best match (mode general/music/speech; arch 1dunet/stft) from `saved_models/FM/FinalModels` or user-specified `--model`.
- Tools used:
	- SDR#: tunes FM band and outputs audio.
	- VB-CABLE: virtual audio device bridging SDR# to Python.
	- sounddevice: real-time capture/playback at 44.1 kHz.
	- Torch: model inference on CPU or CUDA.
	- Optional live plots: waveform + spectrum monitor.
- Settings to check:
	- SDR#: set audio output to “CABLE Input (VB-Audio Virtual Cable)”, 44.1 kHz.
	- live_denoise args: `--input-device "CABLE Output"`, optional `--output-device <speakers>`, `--mode {general|music|speech}`, `--architecture {1dunet|stft}`, `--chunk-size 8192`, `--cpu` if no GPU.
- Shutdown: Ctrl+C triggers graceful stop and saves a live session report PNG (waveforms, spectrograms, estimated SNR improvement).

## Qualitative Verdict (observed)
- Model modes: Speech >>> Music >>> General for FM denoising quality in practice.
- Architectures: 1D U-Net >> STFT U-Net (STFT model present but not fully exercised in runs).

## Quick Commands
- Offline: `python src/inference_fm.py <noisy.wav> <denoised.wav>`
- Live: `python src/live_denoise.py --mode speech --architecture 1dunet --input-device "CABLE Output" --chunk-size 8192`

## Project Configuration
- Central settings: [config.py](config.py#L9-L326) (paths, audio settings, training, RTL-SDR, model defaults).
- Alternate config variant: [config/config.py](config/config.py#L9-L326) (same structure, different FM final path mapping).

## Notes
- All audio is processed mono at 44.1 kHz.
- Training and inference expect model/audio lengths divisible by 16 to align with U-Net pooling.
