# AudioScrubber - Full Repository Guide

Real-time and offline FM audio denoising stack combining deep-learning U-Nets, classical denoisers, and an interactive PyQt5 UI. Designed for SDR captures (RTL-SDR/SDR#/VB-CABLE) at 44.1 kHz.

---

## Table of Contents
- What This Project Does
- High-Level Architecture
- Live Denoising UI (PyQt5)
- Command-Line Tools (Live & Offline)
- Models (Deep Learning) and Classical Denoisers
- Data, Noise, and Chunking
- Training Pipeline
- Configuration
- Repository Map
- Results & Reports
- Troubleshooting & Tips
- References & Further Reading

---

## What This Project Does
- Cleans noisy FM audio streams in real time (SDR# -> VB-CABLE -> AudioScrubber -> speakers).
- Batch/offline denoising of files or folders with optional visual reports.
- Side-by-side algorithm comparison, including a Mass Compare mode that auto-runs multiple denoisers and saves consolidated plots/CSVs/JSON (white-background plots).
- Supports both deep-learning U-Nets (waveform and STFT) and classical methods (spectral subtraction, Wiener, adaptive MMSE/STSA).

---

## High-Level Architecture
- **Front-end UI**: [src/ui/live_denoiser_app.py](src/ui/live_denoiser_app.py) - PyQt5 dashboard with tabs for live view, SNR comparison, algorithm comparison, performance metrics, and Mass Compare automation.
- **Audio engine**: queue-based worker inside the UI (sounddevice I/O, denoiser dispatch, metrics, plotting hooks).
- **Denoiser registry**: [src/fm/classical/__init__.py](src/fm/classical/__init__.py) and [src/fm/model_loader.py](src/fm/model_loader.py) register classical and DL denoisers for discovery in UI/CLI.
- **Deep-learning models**: waveform 1D U-Net and STFT 2D U-Net under [src/fm/neuralnets](src/fm/neuralnets), served by model loader and training scripts.
- **Classical denoisers**: spectral subtraction, standard/adaptive Wiener, and MMSE-STSA variants in [src/fm/classical](src/fm/classical).
- **CLI tools**: live runner [src/live_denoise.py](src/live_denoise.py), offline runner [src/inference_fm.py](src/inference_fm.py), inference orchestrator [src/inference/main_inference.py](src/inference/main_inference.py).

---

## Live Denoising UI (PyQt5)
File: [src/ui/live_denoiser_app.py](src/ui/live_denoiser_app.py)

### Tabs
- Live View: waveform and spectrum for noisy vs denoised audio, device selection, start/stop.
- SNR Comparison: live SNR tracking before/after denoising.
- Algorithm Comparison: run selected algorithms side-by-side on a short capture.
- Performance: per-chunk processing time, CPU/GPU/memory, dropped chunks.
- Mass Compare: auto-runs multiple algorithms sequentially (default 30 s each), plots all traces, saves PNG/CSV/JSON automatically with white backgrounds.

### Key Behaviors
- sounddevice stream at 44.1 kHz; queue-based AI thread; fallback to last output to avoid gaps.
- Dynamic denoiser switching (deep-learning or classical) via combo box.
- Baseline SNR tracked; per-algorithm SNR/latency recorded during Mass Compare.
- Auto-save of Mass Compare outputs to results/mass_compare/:
  - SNR comparison plot (white background PNG)
  - Results CSV (per algorithm metrics)
  - Detailed JSON (raw traces, summary: best SNR, fastest)

### Quick Start (UI)
1) Start SDR# and set audio output to CABLE Input (VB-Audio Virtual Cable) at 44.1 kHz.
2) In AudioScrubber, select input device CABLE Output and your speakers as output.
3) Pick a denoiser (DL speech/music/general or classical) and click Start.
4) Use Mass Compare tab to batch-test algorithms and auto-save graphs.

---

## Command-Line Tools

### Live CLI
File: [src/live_denoise.py](src/live_denoise.py)

Run real-time denoising headless:
```bash
python src/live_denoise.py --mode speech --architecture 1dunet --input-device "CABLE Output" --chunk-size 8192
```
Options: --output-device, --cpu, --model <path>, --plot (enable live plots), --passthrough (bypass model).

### Offline CLI
File: [src/inference_fm.py](src/inference_fm.py)

Denoise a file or folder and emit report PNG:
```bash
python src/inference_fm.py noisy.wav denoised.wav
```
Uses default models from config; chunked inference with padding to training length.

### Inference Orchestrator
File: [src/inference/main_inference.py](src/inference/main_inference.py)

Batch utilities and plotting helpers for inference experiments.

---

## Models (Deep Learning)
- Waveform 1D U-Net: [src/fm/neuralnets/neuralnet.py](src/fm/neuralnets/neuralnet.py) - encoder/decoder with skips on raw audio.
- STFT 2D U-Net: [src/fm/neuralnets/stft_unet2d.py](src/fm/neuralnets/stft_unet2d.py) - magnitude-spectrogram model.
- Loader/registry: [src/fm/model_loader.py](src/fm/model_loader.py) - infers architecture/mode, restricts to compatible FinalModels, exposes DLDenoiserWrapper for UI.
- Weights: [saved_models/FM/FinalModels](saved_models/FM/FinalModels) - speech/music/general for 1D U-Net and STFT; plus checkpoints under saved_models/FM/models.

### Classical Denoisers
Folder: [src/fm/classical](src/fm/classical)
- [spectral_subtraction.py](src/fm/classical/spectral_subtraction.py): Boll 1979, oversubtraction, multiband option, spectral flooring.
- [wiener_filter.py](src/fm/classical/wiener_filter.py): optimal linear MMSE Wiener with noise PSD estimation.
- [adaptive_wiener.py](src/fm/classical/adaptive_wiener.py): decision-directed a priori SNR + MMSE-STSA (Ephraim & Malah) with adaptive noise tracking.
- Registry in [__init__.py](src/fm/classical/__init__.py) exposes multiple variants (standard, adaptive, multiband, conservative/agg modes).

---

## Data, Noise, and Chunking
- Clean speech: LibriSpeech dev-clean/dev-other resampled to 44.1 kHz under [dataset/LibriSpeech](dataset/LibriSpeech).
- Music: curated mono clips under [dataset/instant](dataset/instant).
- Noise: real FM noise sweeps in [dataset/noise](dataset/noise).
- Training segment length: 88,192 samples (~2 s), divisible by 16 for 4 U-Net downsamples.
- Live chunk: 8,192 samples (~186 ms) default; 4,096 fallback (configurable in [config.py](config.py)).
- On-the-fly mixing during training: random crop/pad of clean + random noise slice + random SNR per sample (see [src/fm/1dunet/backshot.py](src/fm/1dunet/backshot.py)). Keeps memory low and diversifies examples.

---

## Training Pipeline (1D U-Net)
File: [src/fm/1dunet/backshot.py](src/fm/1dunet/backshot.py)
- Streams LibriSpeech clean audio; lazily loads FM noise.
- Random SNR mixing per example; segment length 88,192.
- Defaults: Adam LR 1e-4, batch 2–4, epochs 50–100, grad clip 1.0, ReduceLROnPlateau, checkpoints every 5 epochs, best-on-val save.
- CUDA auto-detect; small batches to manage memory.

---

## Configuration
- Primary config: [config.py](config.py)  paths, audio params, training defaults, RTL-SDR/VB-CABLE settings, model defaults, chunk sizes.
- Alt config: [config/config.py](config/config.py)  same structure, different FM model path mapping.

---

## Repository Map (selected)
- [config.py](config.py) and [config/config.py](config/config.py): global settings.
- [src/ui/live_denoiser_app.py](src/ui/live_denoiser_app.py): PyQt5 UI and audio engine.
- [src/live_denoise.py](src/live_denoise.py): CLI live runner.
- [src/inference_fm.py](src/inference_fm.py): offline denoising.
- [src/fm/model_loader.py](src/fm/model_loader.py): DL model discovery and wrapping.
- [src/fm/neuralnets](src/fm/neuralnets): 1D/2D U-Net definitions.
- [src/fm/classical](src/fm/classical): classical denoisers and registry.
- [docs](docs): FM deep-dive, Mermaid diagrams, IEEE-style report draft.
- [saved_models/FM/FinalModels](saved_models/FM/FinalModels): shipped checkpoints.
- [results](results): saved reports and Mass Compare outputs.

---

## Results & Reports
- Offline and live runs can emit PNG reports (waveform, spectrogram, estimated SNR gain).
- Mass Compare auto-saves to results/mass_compare/ with PNG (white background), CSV, and JSON summaries.

---

## Troubleshooting & Tips
- Ensure SDR# outputs to CABLE Input and Python listens to CABLE Output at 44.1 kHz.
- If GPU unavailable or unstable, add --cpu in live CLI or select CPU in UI; classical denoisers are CPU-light.
- Chunk underruns: reduce chunk size or disable heavy models; fallback replays last chunk to mask gaps.
- Model mismatch errors: use weights from saved_models/FM/FinalModels (UI loader already restricts here).
- Security warning from torch.load: models are trusted local checkpoints; you may set weights_only=True if you modify loader for untrusted files.

---

## References & Further Reading
- Detailed FM guide: [docs/FM_README.md](docs/FM_README.md)
- IEEE-style report draft: [docs/FM_IEEE_Report.md](docs/FM_IEEE_Report.md)
- Mermaid runtime diagrams: [docs/live_denoise_mermaid.md](docs/live_denoise_mermaid.md)
