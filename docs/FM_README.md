# AudioScrubber FM Denoising — Deep Dive

This document is the detailed companion to the main Readme. It covers hardware, datasets, noise sourcing, chunking rationale, model architectures, training and inference flows, and qualitative verdicts for FM denoising.

## Hardware and Signal Chain
- Receiver: RTL-SDR USB dongle (primary). Airspy or similar SDRs can be used if they emit PCM audio at 44.1 kHz.
- Tuning app: SDR# to lock on FM stations and stream audio.
- Virtual audio bridge: VB-CABLE. SDR# output device set to "CABLE Input"; Python listens to "CABLE Output".
- Optional: Speakers/headphones as output device; GPU (CUDA) if available for lower latency.

## Datasets and Noise Construction
- Speech: LibriSpeech dev-clean and dev-other, resampled to 44.1 kHz, mono FLAC/WAV (see paths in config.py and config/config.py).
- Music: Curated mono music clips at 44.1 kHz under dataset/music.
- Noise capture: FM noise recorded by sweeping/scouring the FM band; stored as dataset/noise/FM_Noise.wav. This is spectrum-sourced interference, not synthetic white noise.
- Modes / model families:
  - Speech model: trained on LibriSpeech speech with FM noise.
  - Music model: trained on music clips with FM noise.
  - General model: broader mix; quality trails speech/music on FM tests.
- On-the-fly mixing (backshot dataloader):
  - Each clean clip is cropped/padded to 88,192 samples (~2 s at 44.1 kHz).
  - Noise is lazily loaded once, then random segments are mixed at random SNRs per sample.
  - Benefits: memory-light, more variety, no pre-generated noisy dataset. See src/fm/1dunet/backshot.py.

## Chunking Methodology
- Training segment length: 88,192 samples (~2 s), divisible by 16 to align with 4 U-Net downsamplings. Longer context reduces boundary artifacts and helps the network model temporal structure.
- Live chunk size: 8,192 samples by default (~186 ms). Chosen as a compromise between latency and context; large enough for denoising quality, small enough for real-time responsiveness. A 4,096-sample fallback exists in config.py for lower-latency tests.
- Gap avoidance: live_denoise reuses the last output chunk if the AI queue is empty, preventing audible dropouts.

## Models and Architectures
- 1D U-Net (waveform):
  - Source: src/fm/neuralnets/neuralnet.py.
  - Encoder-decoder with skip connections; operates directly on time-domain audio.
  - Used in training script src/fm/1dunet/backshot.py and live/inference flows.
- STFT 2D U-Net (spectrogram):
  - Source: src/fm/neuralnets/stft_unet2d.py.
  - Processes magnitude spectrograms; alternative architecture. Present but not fully exercised in runs.
- Model loader and discovery: src/fm/model_loader.py auto-detects architecture (1dunet/stft) and mode (speech/music/general) from saved_models/FM/FinalModels.
- Exported weights:
  - 1D U-Net: saved_models/FM/FinalModels/FM_Final_1DUNET/{general,music,speech}.pth
  - STFT U-Net: saved_models/FM/FinalModels/FM_Final_STFT/{general,music,speech}.pth

## Training Flow (1D U-Net)
- Script: src/fm/1dunet/backshot.py.
- Data loader: streams LibriSpeech clean audio; mixes FM_Noise.wav on-the-fly with random SNR.
- Hyperparameters (defaults): LR 1e-4 (Adam/AdamW), batch 2–4, epochs 50–100, gradient clip 1.0, ReduceLROnPlateau scheduler, checkpoints every 5 epochs, best-model saving on val loss.
- Memory aids: lazy noise load, streaming reads, small batch, CUDA auto-detect.

## Offline Inference Flow
- Script: src/inference_fm.py.
- Inputs: a single file or a folder of FLAC/WAV.
- Processing: chunked to training length; padding applied; per-chunk denoise with 1D U-Net; concatenation back to full audio.
- Outputs: denoised audio + report PNG (waveforms, spectrograms, estimated SNR gain).
- Defaults: 44.1 kHz sample rate; model path from config or specified path.

## Real-Time Pipeline (Step-by-Step)
1) SDR# tunes FM and outputs audio to "CABLE Input" (VB-CABLE).
2) live_denoise.py opens sounddevice stream: input "CABLE Output" -> output speakers.
3) Audio thread grabs chunks (8,192 samples) and enqueues them.
4) AI thread dequeues, denoises with the selected model (mode speech/music/general; arch 1dunet/stft), re-enqueues clean audio.
5) Audio thread plays cleaned chunks; if the queue is empty, it reuses the last output chunk to avoid gaps.
6) Optional live plot shows waveform and spectrum.
7) On exit (Ctrl+C), a report PNG is generated from the last ~10 seconds (waveforms, spectrograms, estimated SNR improvement).

### Tools in the loop
- RTL-SDR/Airspy: RF front-end for FM.
- SDR#: tuning UI and demodulator.
- VB-CABLE: virtual audio bridge.
- sounddevice: low-latency capture/playback at 44.1 kHz.
- PyTorch: model inference on CPU or CUDA.

### Key settings to verify
- SDR#: audio output set to "CABLE Input", sample rate 44.1 kHz.
- live_denoise.py: --input-device "CABLE Output", optional --output-device <speakers>, --mode {speech|music|general}, --architecture {1dunet|stft}, --chunk-size 8192, add --cpu if no GPU.

## Qualitative Verdicts (observed)
- Mode quality: Speech >>> Music >>> General for FM denoising.
- Architecture: 1D U-Net >> STFT U-Net (STFT present but not run in depth).

## Quick Commands
- Offline: python src/inference_fm.py <noisy.wav> <denoised.wav>
- Live: python src/live_denoise.py --mode speech --architecture 1dunet --input-device "CABLE Output" --chunk-size 8192

## References in Repo
- Configs: config.py, config/config.py (paths, audio params, training, RTL-SDR).
- Training: src/fm/1dunet/backshot.py
- Models: src/fm/neuralnets/neuralnet.py, src/fm/neuralnets/stft_unet2d.py
- Model loader: src/fm/model_loader.py
- Inference: src/inference_fm.py
- Live: src/live_denoise.py

## Future Enhancements
- Quantitative eval: PESQ/STOI/segmental SNR on held-out FM captures to rank 1D vs STFT and speech vs music vs general.
- Latency tuning: overlap-add for smoother live output; profile chunk sizes per device.
- Data enrichment: more real FM noise sweeps and air-recorded speech/music for robustness.
- Packaging: prebuilt CLI examples and a minimal GUI for device selection and live monitoring.
