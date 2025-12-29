# FM Radio Denoising Using Deep Learning Techniques

Satya Srinivas Paladugu, Akshita Dindukurthi, Prithvi S, Ekansh Khullar, Dr. Sirisha Tadepalli  
Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Bengaluru, India

Emails: bl.ai.ai4ai24063@bl.students.amrita.edu, bl.ai.ai4ai24004@bl.students.amrita.edu, bl.ai.ai4ai24076@bl.students.amrita.edu, bl.ai.ai4ai24075@bl.students.amrita.edu, t_sirisha@blr.amrita.edu

---

## Abstract
We present AudioScrubber, a deep learning pipeline for denoising frequency-modulated (FM) radio audio captured with commodity software-defined radios (SDRs). The system combines spectrum-sourced FM noise with clean speech and music datasets to train waveform and spectrogram U-Net models. A streaming inference stack integrates SDR#, VB-CABLE, and a PyTorch runtime to deliver real-time denoising at 44.1 kHz with minimal latency. Empirically, speech-focused 1D U-Net models outperform music-specific and general models; 1D U-Net also surpasses the STFT-based variant in subjective quality. This report details the data sourcing, on-the-fly noise mixing, model architectures, training regime, and live inference design.

**Index Terms**—FM denoising, software-defined radio, U-Net, speech enhancement, real-time audio, VB-CABLE, SDR#.

## I. Introduction
Reliable FM audio remains challenging in urban RF environments due to multipath fading, adjacent-channel interference, and front-end noise. Traditional analog filtering suppresses narrowband interference but struggles with broadband or rapidly varying noise. Deep learning denoisers can model nonstationary artifacts directly in the time-frequency domain, offering improved perceptual quality for speech and music.

This work targets practical FM listening and downstream speech tasks (e.g., ASR) using low-cost RTL-SDR hardware. We record real FM noise by sweeping the band and pair it with clean LibriSpeech speech and curated music. Noise is injected on-the-fly during training at randomized SNRs, avoiding large pre-generated corpora. The primary model is a 1D U-Net operating on waveforms; an alternative STFT 2D U-Net is provided for frequency-domain experiments.

A real-time pipeline couples SDR# (tuning/demodulation) with VB-CABLE (virtual audio routing) and a PyTorch inference engine (sounddevice I/O). Chunks of 8,192 samples (~186 ms at 44.1 kHz) balance latency and context; fallback buffering prevents gaps if the AI thread lags. Best-performing exports reside under saved_models/FM/FinalModels (speech, music, general). Observed quality ranks Speech > Music > General, with 1D U-Net outperforming the STFT variant.

The remainder of this report (to be completed) will cover related work, data preparation, model design, training details, evaluation, and ablation findings for FM denoising.
