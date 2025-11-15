# Quick Start Guide - SNR-Based Training

## ✅ What's Been Set Up

### 1. **SNR-Based Noise Addition (AWN2.py)**
- Industry-standard Signal-to-Noise Ratio approach
- Default: 0, 5, 10, 15, 20 dB noise levels
- Works for both speech and music
- Location: `src/dataset_creation/AWN2.py`

### 2. **New Directory Structure**
```
dataset/
├── LibriSpeech_processed_snr/   # NEW: SNR-based speech dataset
│   ├── clean/
│   └── noisy/
├── music/                        # NEW: Music dataset
│   ├── raw/                      # Put your music files here
│   └── music_processed/
│       ├── clean/
│       └── noisy/
└── LibriSpeech_processed/        # OLD: Keep for reference
```

### 3. **Separate Training Scripts**
- `src/model/backshot.py` → **Speech model** (22050 Hz)
- `src/model/backshot_music.py` → **Music model** (44100 Hz)

### 4. **Updated Configs**
- `config.py` → Speech configuration (updated for SNR dataset)
- `config_music.py` → Music configuration (44.1kHz, 88192 samples)

---

## 🚀 Step-by-Step Workflow

### For Speech (LibriSpeech)

**Step 1: Process Dataset**
```bash
python src/dataset_creation/AWN2.py \
    --input dataset/LibriSpeech/dev-clean \
    --output dataset/LibriSpeech_processed_snr \
    --snr_levels 0 5 10 15 20
```

**Step 2: Train Model**
```bash
# Fresh training
python src/model/backshot.py

# Resume if interrupted
python src/model/backshot.py resume
```

**Output:** `saved_models/unet1d_best.pth`

---

### For Music

**Step 1: Add Music Files**
```bash
# Copy your music files to:
dataset/music/raw/
```

**Step 2: Process Dataset**
```bash
python src/dataset_creation/AWN2.py \
    --input dataset/music/raw \
    --output dataset/music_processed \
    --snr_levels 0 5 10 15 20 \
    --sample_rate 44100
```

**Step 3: Train Model**
```bash
# Fresh training
python src/model/backshot_music.py

# Resume if interrupted
python src/model/backshot_music.py resume
```

**Output:** `saved_models/unet1d_music_best.pth`

---

## 📊 Key Differences: Speech vs Music

| Feature | Speech (backshot.py) | Music (backshot_music.py) |
|---------|---------------------|--------------------------|
| Sample Rate | 22050 Hz | 44100 Hz (CD quality) |
| Audio Length | 44096 samples (~2s) | 88192 samples (~2s) |
| Dataset Path | LibriSpeech_processed_snr | music_processed |
| Model Output | unet1d_best.pth | unet1d_music_best.pth |
| Checkpoint Dir | checkpoints/ | music_checkpoints/ |
| Use Case | Speech, podcasts | Music, songs |

---

## 🎯 SNR Levels Explained

```
SNR (dB) = 10 × log₁₀(signal_power / noise_power)
```

| SNR | Description | Use Case |
|-----|-------------|----------|
| 0 dB | Very noisy (signal = noise) | Extreme conditions |
| 5 dB | Noisy (signal 3× noise) | Noisy environments |
| 10 dB | Moderate (signal 10× noise) | Typical real-world |
| 15 dB | Light noise (signal 32× noise) | Clean with artifacts |
| 20 dB | Very clean (signal 100× noise) | Studio quality |

**Why 5 levels?** Provides good coverage without excessive data augmentation.

---

## ⏱️ Time Estimates (8-Hour Training Window)

### Speech Processing
- **Dataset processing:** ~30-60 min (one-time)
- **Training:** ~8 hours max (40 epochs @ 12 min/epoch)
- **Total first run:** ~9 hours

### Music Processing
- **Dataset processing:** Depends on music dataset size
- **Training:** ~8 hours max (same as speech)

---

## 📋 Checklist

### Before Training Speech Model
- [ ] LibriSpeech dev-clean is in `dataset/LibriSpeech/dev-clean/`
- [ ] Run AWN2.py to create processed dataset
- [ ] Verify output: clean (5,567) + noisy (27,835) files
- [ ] Start training with `python src/model/backshot.py`

### Before Training Music Model
- [ ] Music files are in `dataset/music/raw/`
- [ ] Run AWN2.py with `--sample_rate 44100`
- [ ] Verify output structure
- [ ] Start training with `python src/model/backshot_music.py`

---

## 🔧 Common Commands

### Preview Dataset Processing
```bash
# See what will be created without actually processing
python src/dataset_creation/AWN2.py --input <input_dir> --output <output_dir> --preview
```

### Check Training Progress
```bash
# View checkpoints
ls saved_models/checkpoints/          # Speech
ls saved_models/music_checkpoints/    # Music

# View models
ls saved_models/*.pth
```

### Resume Training
```bash
# Speech
python src/model/backshot.py resume

# Music
python src/model/backshot_music.py resume
```

---

## 📁 File Organization

```
AudioScrubber/
├── src/
│   ├── dataset_creation/
│   │   └── AWN2.py                    # NEW: SNR-based noise
│   └── model/
│       ├── backshot.py                # UPDATED: Uses SNR dataset
│       └── backshot_music.py          # NEW: Music training
├── config.py                          # UPDATED: SNR paths
├── config_music.py                    # NEW: Music config
├── DATASET_PREP_GUIDE.md             # NEW: Full guide
├── TRAINING_GUIDE.md                 # Existing guide
└── dataset/
    ├── LibriSpeech_processed_snr/    # NEW: SNR-based speech
    └── music/                         # NEW: Music datasets
```

---

## 🎉 You're Ready!

### Next Actions:

1. **Process LibriSpeech with SNR noise:**
   ```bash
   python src/dataset_creation/AWN2.py \
       --input dataset/LibriSpeech/dev-clean \
       --output dataset/LibriSpeech_processed_snr
   ```

2. **Start speech training:**
   ```bash
   python src/model/backshot.py
   ```

3. **When you have music data:**
   - Copy to `dataset/music/raw/`
   - Process with AWN2.py (44100 Hz)
   - Train with `backshot_music.py`

---

## 📚 Documentation

- **Full dataset guide:** `DATASET_PREP_GUIDE.md`
- **Training guide:** `TRAINING_GUIDE.md`
- **Music config:** `config_music.py`
- **AWN2 help:** `python src/dataset_creation/AWN2.py --help`

Good luck with your training! 🚀
