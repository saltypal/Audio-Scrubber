# Dataset Preparation Guide

## Overview

This guide explains how to prepare datasets for both **speech** and **music** audio denoising models using the SNR-based noise addition method (AWN2.py).

## 📊 SNR-Based Noise Addition (Industry Standard)

**AWN2.py** uses Signal-to-Noise Ratio (SNR) in decibels (dB) to add controlled noise:

```
SNR (dB) = 10 × log₁₀(signal_power / noise_power)
```

**Default SNR Levels:** 0, 5, 10, 15, 20 dB
- **0 dB**: Equal signal and noise power (very noisy)
- **5 dB**: Signal 3× stronger than noise (noisy)
- **10 dB**: Signal 10× stronger than noise (moderate)
- **15 dB**: Signal 32× stronger than noise (light noise)
- **20 dB**: Signal 100× stronger than noise (very clean)

---

## 🗣️ Speech Dataset Preparation (LibriSpeech)

### Step 1: Verify LibriSpeech Dataset

```bash
# Your LibriSpeech data should be at:
dataset/LibriSpeech/dev-clean/
```

**Structure:**
```
dataset/LibriSpeech/dev-clean/
├── 1272/
│   ├── 128104/
│   │   ├── 1272-128104-0000.flac
│   │   ├── 1272-128104-0001.flac
│   │   └── ...
│   └── 135031/
├── 1462/
└── ...
```

### Step 2: Process with AWN2.py

```bash
# Preview processing (see what will happen)
python src/dataset_creation/AWN2.py \
    --input dataset/LibriSpeech/dev-clean \
    --output dataset/LibriSpeech_processed_snr \
    --preview

# Process dataset with SNR-based noise
python src/dataset_creation/AWN2.py \
    --input dataset/LibriSpeech/dev-clean \
    --output dataset/LibriSpeech_processed_snr \
    --snr_levels 0 5 10 15 20 \
    --sample_rate 22050
```

**Expected Output:**
```
dataset/LibriSpeech_processed_snr/
├── clean/
│   └── 1272/
│       └── 128104/
│           ├── 1272-128104-0000.flac
│           ├── 1272-128104-0001.flac
│           └── ... (5,567 files)
└── noisy/
    └── 1272/
        └── 128104/
            ├── 1272-128104-0000_snr_0dB.flac
            ├── 1272-128104-0000_snr_5dB.flac
            ├── 1272-128104-0000_snr_10dB.flac
            ├── 1272-128104-0000_snr_15dB.flac
            ├── 1272-128104-0000_snr_20dB.flac
            └── ... (27,835 files = 5,567 × 5 SNR levels)
```

### Step 3: Train Speech Model

```bash
# Train speech denoiser (8 hours max)
python src/model/backshot.py

# Resume if interrupted
python src/model/backshot.py resume
```

**Model saved to:** `saved_models/unet1d_best.pth`

---

## 🎵 Music Dataset Preparation

### Step 1: Organize Music Dataset

Place your music files in the raw directory:

```bash
# Create directory structure
mkdir -p dataset/music/raw

# Copy your music files here
# Supported formats: .flac, .wav, .mp3
```

**Example structure:**
```
dataset/music/raw/
├── song1.flac
├── song2.wav
├── album1/
│   ├── track01.flac
│   ├── track02.flac
│   └── ...
└── album2/
    ├── track01.mp3
    └── ...
```

### Step 2: Process Music with AWN2.py

```bash
# Preview processing
python src/dataset_creation/AWN2.py \
    --input dataset/music/raw \
    --output dataset/music_processed \
    --sample_rate 44100 \
    --preview

# Process music dataset (CD quality: 44.1kHz)
python src/dataset_creation/AWN2.py \
    --input dataset/music/raw \
    --output dataset/music_processed \
    --snr_levels 0 5 10 15 20 \
    --sample_rate 44100
```

**Why 44100 Hz for music?**
- Speech: 22050 Hz is sufficient (most energy < 8 kHz)
- Music: 44100 Hz preserves full audible spectrum (20-20000 Hz)

**Expected Output:**
```
dataset/music_processed/
├── clean/
│   ├── song1.flac
│   ├── song2.flac
│   └── ...
└── noisy/
    ├── song1_snr_0dB.flac
    ├── song1_snr_5dB.flac
    ├── song1_snr_10dB.flac
    ├── song1_snr_15dB.flac
    ├── song1_snr_20dB.flac
    ├── song2_snr_0dB.flac
    └── ...
```

### Step 3: Train Music Model

```bash
# Train music denoiser (8 hours max)
python src/model/backshot_music.py

# Resume if interrupted
python src/model/backshot_music.py resume
```

**Model saved to:** `saved_models/unet1d_music_best.pth`

---

## 📋 Quick Reference Commands

### Speech Processing & Training

```bash
# 1. Process LibriSpeech
python src/dataset_creation/AWN2.py \
    --input dataset/LibriSpeech/dev-clean \
    --output dataset/LibriSpeech_processed_snr \
    --snr_levels 0 5 10 15 20

# 2. Train speech model
python src/model/backshot.py

# 3. Resume if needed
python src/model/backshot.py resume
```

### Music Processing & Training

```bash
# 1. Process music dataset
python src/dataset_creation/AWN2.py \
    --input dataset/music/raw \
    --output dataset/music_processed \
    --snr_levels 0 5 10 15 20 \
    --sample_rate 44100

# 2. Train music model
python src/model/backshot_music.py

# 3. Resume if needed
python src/model/backshot_music.py resume
```

---

## 🔍 Verification

### Check Dataset Size

```bash
# Count clean files
ls dataset/LibriSpeech_processed_snr/clean/**/*.flac | wc -l

# Count noisy files
ls dataset/LibriSpeech_processed_snr/noisy/**/*.flac | wc -l
# Should be 5× the clean count

# For music
ls dataset/music_processed/clean/*.flac | wc -l
ls dataset/music_processed/noisy/*.flac | wc -l
```

### Verify Audio Quality

```python
# Test loading processed audio
import librosa
import soundfile as sf

# Load clean and noisy pair
clean, sr = librosa.load('dataset/LibriSpeech_processed_snr/clean/1272/128104/1272-128104-0000.flac')
noisy, sr = librosa.load('dataset/LibriSpeech_processed_snr/noisy/1272/128104/1272-128104-0000_snr_10dB.flac')

print(f"Clean: shape={clean.shape}, sr={sr}")
print(f"Noisy: shape={noisy.shape}, sr={sr}")
```

---

## 📊 Dataset Statistics

### Expected File Counts

| Dataset | Clean Files | SNR Levels | Noisy Files | Total |
|---------|-------------|------------|-------------|-------|
| LibriSpeech dev-clean | 5,567 | 5 | 27,835 | 33,402 |
| Music (example) | 1,000 | 5 | 5,000 | 6,000 |

### Storage Requirements

| Dataset | Sample Rate | Duration/File | Clean Size | Noisy Size | Total |
|---------|-------------|---------------|------------|------------|-------|
| Speech | 22050 Hz | ~2-10s | ~500 MB | ~2.5 GB | ~3 GB |
| Music | 44100 Hz | ~30-180s | Varies | 5× clean | Varies |

---

## 🎯 Training Timeline (8 Hours)

### Speech Model
```
Sample rate: 22050 Hz
Audio length: 44096 samples (~2s)
Batch size: 16
Estimated: ~12 min/epoch × 40 epochs = 8 hours
```

### Music Model
```
Sample rate: 44100 Hz
Audio length: 88192 samples (~2s)
Batch size: 16
Estimated: ~12 min/epoch × 40 epochs = 8 hours
```

---

## ⚠️ Common Issues

### Issue: "No audio files found"
**Solution:** Check input directory path and file extensions

### Issue: "GPU out of memory"
**Solution:** Reduce batch size in config
```python
# In config.py or config_music.py
BATCH_SIZE = 8  # Reduce from 16 to 8
```

### Issue: "Checkpoint not found"
**Solution:** Train for at least 1 epoch before resuming

### Issue: Different audio lengths
**Solution:** AWN2.py preserves original lengths. Model handles padding/truncating automatically.

---

## 🚀 Next Steps

After dataset preparation:

1. **Verify dataset:** Run preview mode to check file counts
2. **Start training:** Use appropriate backshot script
3. **Monitor progress:** Check loss decreasing in first few epochs
4. **Resume if interrupted:** Use resume command
5. **Evaluate model:** Test on held-out samples after training

---

## 📝 Notes

- **SNR-based noise is industry standard** for audio research
- **Same AWN2.py works for speech and music** (just change sample rate)
- **5 SNR levels provide good coverage** (0-20 dB range)
- **Preprocessing is one-time** (save processed dataset for multiple experiments)
- **Resume feature saves time** (can stop/start training anytime)

Happy training! 🎉
