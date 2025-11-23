# AudioScrubber Project Structure

## Directory Organization

```
AudioScrubber/
├── config/                    # Configuration files
│   ├── config.py             # Main system configuration
│   └── config_music.py       # Music-specific configuration
│
├── data/                      # Data directory
│   └── raw/                  # Raw data files
│       └── captured_ofdm.iq  # Captured OFDM IQ samples
│
├── dataset/                   # Training datasets
│   ├── instant/              # Quick test samples
│   │   ├── clean/
│   │   └── noisy/
│   ├── LibriSpeech/          # Speech dataset
│   │   ├── dev-clean/
│   │   └── dev-other/
│   ├── noise/                # Noise samples for training
│   └── OFDM/                 # OFDM dataset
│       ├── clean_ofdm.iq
│       └── noisy_ofdm.iq
│
├── docs/                      # Documentation
│   ├── AI_MODEL_VERIFICATION.md
│   ├── DATASET_PREP_GUIDE.md
│   ├── QUICKSTART_SNR.md
│   ├── REFACTORING_SUMMARY.md
│   ├── RTL_SDR_README.md
│   ├── STATUS_CHECK.md
│   ├── STFT_MODEL_INFO.md
│   └── TRAINING_GUIDE.md
│
├── output/                    # Generated outputs
│   ├── dataset/              # Dataset generation outputs
│   ├── general/              # General denoising outputs
│   ├── music/                # Music processing outputs
│   ├── music_denoised/       # Music denoising results
│   └── speech/               # Speech processing outputs
│
├── results/                   # Test results and reports
│   ├── live_report_*.png     # Live denoising reports
│   ├── SONG_PLAYING_*.png    # Song processing visualizations
│   └── new_lib_test.png      # Library test results
│
├── saved_models/              # Trained model checkpoints
│   ├── FM/                   # FM denoising models
│   │   ├── FM_Final_1DUNET/
│   │   └── models/
│   └── OFDM/                 # OFDM models
│       ├── final_models/
│       └── test_output/
│
├── scripts/                   # Utility and test scripts
│   ├── test_refactoring.py   # Refactoring tests
│   ├── test_sdr.py           # SDR testing
│   ├── troubleshoot_rtlsdr.py # RTL-SDR troubleshooting
│   └── validate_system.py    # System validation
│
├── src/                       # Source code
│   ├── fm/                   # FM denoising module
│   │   ├── fm_record/        # FM recording utilities
│   │   ├── model/            # Neural network models
│   │   └── model2/           # Alternative model implementations
│   │
│   ├── ofdm/                 # OFDM module
│   │   ├── lib/              # Core OFDM library
│   │   │   ├── __init__.py
│   │   │   ├── config.py     # OFDM configuration
│   │   │   ├── core.py       # OFDM engine (FFT/IFFT)
│   │   │   ├── modulation.py # Modulation schemes (QPSK)
│   │   │   ├── receiver.py   # Channel equalization
│   │   │   ├── transceiver.py # Transmitter/Receiver
│   │   │   └── test_system.py # System tests
│   │   │
│   │   ├── autoencoder/      # Autoencoder approach
│   │   ├── model/            # OFDM neural models
│   │   ├── TxRx/             # Transmission/Reception
│   │   └── dataset_gnu.py    # GNU Radio dataset generation
│   │
│   ├── inference_fm.py       # FM inference script
│   └── live_denoise.py       # Real-time denoising
│
├── Tests/                     # Test data and results
│   ├── samples/              # Test audio samples
│   └── tests/                # Test outputs
│
├── .gitignore                # Git ignore rules
├── Readme.md                 # Main project README
├── requirements.txt          # Python dependencies
└── PROJECT_STRUCTURE.md      # This file

```

## Key Components

### 1. **FM Denoising System**
- Models: 1D U-Net architecture
- Specialized models for: General, Music, Speech
- Real-time denoising capability via `live_denoise.py`

### 2. **OFDM Communication System**
- Modular library design (`src/ofdm/lib/`)
- QPSK modulation with channel equalization
- AI-based denoising for OFDM signals
- GNU Radio integration for dataset generation

### 3. **Configuration Management**
- Centralized in `config/` directory
- Separate configs for different audio types
- Easy parameter tuning via dataclasses

### 4. **Documentation**
- Comprehensive guides in `docs/` directory
- Training guides, status checks, and troubleshooting

## Usage Quick Reference

### Running FM Denoising
```bash
python src/inference_fm.py --input <audio_file> --output <output_dir>
python src/live_denoise.py --passthrough  # Real-time mode
```

### Testing OFDM System
```bash
python src/ofdm/lib/test_system.py
```

### System Validation
```bash
python scripts/validate_system.py
python scripts/test_sdr.py
```

## Import Paths
After reorganization, imports should use:
```python
from config.config import Paths, AudioSettings
from config.config_music import Paths, AudioSettings, TrainingConfig
```
