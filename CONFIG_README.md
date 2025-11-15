# Configuration Guide

## Overview
All project settings are centralized in `config.py` for consistency and easy management.

## Quick Start

```python
# In any script, import what you need
from config import Paths, AudioSettings, NoiseSettings, TrainingConfig

# Use paths
model_path = Paths.MODEL_BEST
clean_dir = Paths.INSTANT_CLEAN

# Use settings
sample_rate = AudioSettings.SAMPLE_RATE
noise_levels = NoiseSettings.NOISE_LEVELS
```

## Configuration Classes

### `Paths`
All file and directory paths
```python
Paths.PROJECT_ROOT           # Project root directory
Paths.DATASET_ROOT           # dataset/
Paths.LIBRISPEECH_ROOT       # dataset/LibriSpeech/
Paths.INSTANT_CLEAN          # dataset/instant/clean/
Paths.INSTANT_NOISY          # dataset/instant/noisy/
Paths.MODEL_BEST             # saved_models/unet1d_best.pth
Paths.MODEL_TUNING_DIR       # saved_models/tuning/
```

### `AudioSettings`
Audio processing parameters
```python
AudioSettings.SAMPLE_RATE    # 22050 Hz
AudioSettings.AUDIO_LENGTH   # 44100 samples (2 seconds)
AudioSettings.CHUNK_SIZE     # 4096 samples
AudioSettings.LATENCY_MS     # ~186 ms
```

### `NoiseSettings`
Noise addition for dataset creation
```python
NoiseSettings.NOISE_LEVELS         # [0.005, 0.01, 0.02, 0.05, 0.1]
NoiseSettings.QUICK_NOISE_LEVELS   # [0.01, 0.05]
```

### `TrainingConfig`
Training hyperparameters
```python
TrainingConfig.DEVICE                    # 'cuda' or 'cpu'
TrainingConfig.BATCH_SIZE                # 16
TrainingConfig.LEARNING_RATE             # 0.0001
TrainingConfig.NUM_EPOCHS                # 100
TrainingConfig.EARLY_STOPPING_PATIENCE   # 20
```

### `RTLSDRSettings`
RTL-SDR radio configuration
```python
RTLSDRSettings.FM_FREQUENCY      # 99.5e6 (99.5 MHz)
RTLSDRSettings.SDR_SAMPLE_RATE   # 22050 Hz
```

## Modifying Settings

### Option 1: Edit config.py directly
```python
# In config.py, change default values
class AudioSettings:
    SAMPLE_RATE = 44100  # Changed from 22050
```

### Option 2: Override in your script
```python
from config import TrainingConfig

# Override for this script only
config = TrainingConfig()
config.BATCH_SIZE = 32
config.LEARNING_RATE = 0.0005
```

### Option 3: Pass as arguments
```python
from config import Paths

# Use config defaults
denoiser = Denoiser()  # Uses Paths.MODEL_BEST

# Or override
denoiser = Denoiser(model_path="custom/model.pth")
```

## Initialize Project

Run this to create all directories:
```python
python config.py
```

Or in code:
```python
from config import initialize_project
initialize_project()
```

## Benefits of Centralized Config

✅ **Single source of truth** - Change once, applies everywhere  
✅ **No hardcoded paths** - All paths computed from PROJECT_ROOT  
✅ **Cross-platform** - Path objects handle Windows/Linux differences  
✅ **Type safety** - IDE autocomplete and type checking  
✅ **Documentation** - All settings documented in one place  
✅ **Easy testing** - Override settings for tests without changing code  

## Updated Scripts

All scripts now use `config.py`:
- ✅ `src/model/backshot.py`
- ✅ `src/inference.py`
- ✅ `src/realtime_denoise.py`
- ✅ `src/rtlsdr_denoise.py`
- ✅ `src/dataset_creation/addAWN.py`
- ✅ `src/dataset_creation/process_all_librispeech.py`
- ✅ `src/dataset_creation/process_librispeech_advanced.py`

## Backward Compatibility

The `Config` class maintains compatibility with old scripts:
```python
# Old way (still works)
from config import Config
config = Config()
clean_dir = config.CLEAN_AUDIO_DIR

# New way (recommended)
from config import Paths
clean_dir = Paths.INSTANT_CLEAN
```
