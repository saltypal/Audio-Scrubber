# RTL-SDR Audio Toolkit

A complete suite of tools for capturing, denoising, and monitoring FM radio using RTL-SDR USB dongles and AI-powered noise reduction.

## 📦 Components

### 1. **RTLSDRCore** (`src/rtlsdr_core.py`)
Core async-based RTL-SDR functionality. Base class used by all other tools.

**Features:**
- Async/await compatible streaming
- FM demodulation (polar discriminator)
- Configurable sample rates and gains
- Easy frequency tuning

**Usage in your code:**
```python
from src.rtlsdr_core import RTLSDRCore
import asyncio

async def main():
    core = RTLSDRCore(frequency=88.5e6, sample_rate=22050)
    await core.initialize()
    async for audio_chunk in core.stream_audio():
        # Process audio_chunk
    await core.close()

asyncio.run(main())
```

---

### 2. **Frequency Recorder** (`src/frequency_recorder.py`)
Record demodulated audio or raw IQ samples from any frequency for later analysis.

**Perfect for:**
- Building noise datasets
- Capturing specific signals
- Signal analysis and debugging

**Record static noise (for dataset collection):**
```bash
python src/frequency_recorder.py --frequency 105.5e6 --duration 60 --output dataset/noise/
```

**Record with raw IQ samples:**
```bash
python src/frequency_recorder.py --frequency 88.5e6 --duration 30 --iq --output dataset/
```

**Parameters:**
- `--frequency`: Frequency in Hz (e.g., 88.5e6 = 88.5 MHz)
- `--duration`: Recording time in seconds
- `--output`: Output directory (default: `dataset/noise/`)
- `--iq`: Save raw IQ samples (optional)
- `--audio`: Save demodulated audio (default)

**Output files:**
- `audio_88.5MHz_20251117_153042.wav` - Demodulated audio
- `iq_88.5MHz_20251117_153042.npy` - Raw IQ samples (numpy format)

---

### 3. **FM Monitor** (`src/fm_monitor.py`)
Real-time FM radio receiver with interactive frequency tuning.

**Features:**
- Live FM radio playback
- Change frequency on the fly
- Adjust gain in real-time
- No files saved (streaming only)

**Start monitoring FM:**
```bash
python src/fm_monitor.py --frequency 99.5e6
```

**Interactive commands:**
```
f <frequency>  - Change frequency (e.g., f 101.1 for 101.1 MHz)
g <gain>       - Change gain in dB (e.g., g 40)
q              - Quit
```

**Example session:**
```
📻 Listening to 99.50 MHz
Commands: f <freq> (tune), g <gain> (gain), q (quit)

f 104.5
✅ Tuned to 104.5 MHz

g 35
✅ Gain set to 35

q
Stopping...
```

---

### 4. **RTL-FM Wrapper** (`src/rtlfm_wrapper.py`)
Alternative tools using the native `rtl_fm` binary (C-based, more efficient).

**Record audio (using rtl_fm binary):**
```bash
python src/rtlfm_wrapper.py record --frequency 105.5e6 --duration 120 --output dataset/noise/
```

**Monitor FM live (using rtl_fm binary):**
```bash
python src/rtlfm_wrapper.py monitor --frequency 99.5e6 --gain 40
```

**Parameters:**
- `--frequency`: Frequency in Hz
- `--duration`: Recording time (record command only)
- `--output`: Output directory
- `--gain`: SDR gain ('auto' or dB value)

---

### 5. **Real-Time Denoiser** (`src/rtlsdr_denoise.py`)
Captures FM radio, denoises with trained AI model, plays clean audio in real-time.

**Listen to denoised FM:**
```bash
# With AI denoising (recommended)
python src/rtlsdr_denoise.py --frequency 99.5e6

# Bypass AI to hear raw signal
python src/rtlsdr_denoise.py --frequency 99.5e6 --no-ai
```

**Parameters:**
- `--frequency`: Target frequency in Hz
- `--device`: `cuda` or `cpu` (default: auto-detect)
- `--gain`: SDR gain ('auto' or dB value)
- `--no-ai`: Bypass AI model (hear raw FM)
- `--model`: Path to trained model checkpoint

---

## 🚀 Quick Start Guide

### Setup
```bash
# Install dependencies
conda install -c conda-forge pyrtlsdr
pip install sounddevice scipy numpy torch soundfile

# Optional: Install rtl-sdr tools for rtlfm_wrapper.py
# Download from: https://github.com/osmocom/rtl-sdr/releases
```

### Windows Setup (First Time)
1. Connect RTL-SDR dongle to USB
2. Run Zadig driver installer (http://zadig.akeo.ie/)
3. Select "RTL2832U" device in Zadig
4. Choose "WinUSB" driver and click "Replace Driver"

### Test RTL-SDR Connection
```bash
python test_sdr.py
# Expected output:
# ✅ SDR object created
# ✅ Received 131072 IQ samples
# ✅ SUCCESS: RTL-SDR is working!
```

---

## 📊 Common Workflows

### Workflow 1: Build Noise Dataset for Model Training
```bash
# Record 5 minutes of static noise at different frequencies
python src/frequency_recorder.py --frequency 105.5e6 --duration 300 --output dataset/noise/
python src/frequency_recorder.py --frequency 107.9e6 --duration 300 --output dataset/noise/
python src/frequency_recorder.py --frequency 95.0e6 --duration 300 --output dataset/noise/

# Files saved to dataset/noise/*.wav
```

### Workflow 2: Find a Good FM Station
```bash
# Start monitor and tune interactively
python src/fm_monitor.py

# Try different frequencies:
f 88.5
f 99.5
f 101.1
f 104.3
# Find one you like, then use that frequency in other tools
```

### Workflow 3: Record and Analyze Raw IQ Data
```bash
# Capture raw IQ at 88.5 MHz
python src/frequency_recorder.py --frequency 88.5e6 --duration 30 --iq --output dataset/

# Load and analyze in Python
import numpy as np
iq_data = np.load('dataset/iq_88.5MHz_20251117_153042.npy')
print(f"Captured {len(iq_data)} IQ samples")
print(f"Duration: {len(iq_data)/240000:.1f} seconds")
```

### Workflow 4: Real-Time Denoising
```bash
# Listen to your favorite station WITH AI denoising
python src/rtlsdr_denoise.py --frequency 99.5e6

# Hear the difference with --no-ai
python src/rtlsdr_denoise.py --frequency 99.5e6 --no-ai
```

---

## 🔧 Troubleshooting

### "rtl_fm.exe not found"
```bash
# Only needed for rtlfm_wrapper.py
# Add C:\x64\ to your Windows PATH environment variable
# Or download from: https://github.com/osmocom/rtl-sdr/releases
```

### "No RTL-SDR device found"
```bash
# Verify USB connection and driver installation
python test_sdr.py
# If fails, reinstall driver with Zadig
```

### Poor audio quality
1. **Move antenna** - Try different positions/lengths
2. **Adjust gain** - Use `g 50` in fm_monitor.py or `--gain 50` in scripts
3. **Change frequency** - Try different FM stations
4. **Use USB 2.0** - Some USB 3.0 ports have EMI issues

### Recording sounds like silence
1. Check speaker volume
2. Verify speaker is plugged in
3. Test audio device: `python -c "import sounddevice as sd; print(sd.default)"`
4. Try `--no-ai` mode to hear raw FM

---

## 📝 Architecture

```
RTLSDRCore (async base class)
    ├── RTLSDRDenoiser (real-time denoising)
    ├── FrequencyRecorder (save to disk)
    ├── FMMonitor (interactive monitoring)
    └── RTLFMWrapper (native rtl_fm binary)
```

All components use the same core RTL-SDR functionality, just with different I/O backends.

---

## 🎯 Next Steps

1. **Test your setup**: `python test_sdr.py`
2. **Listen to FM**: `python src/fm_monitor.py`
3. **Record noise**: `python src/frequency_recorder.py --duration 60 --output dataset/noise/`
4. **Try denoising**: `python src/rtlsdr_denoise.py`

---

## 📞 Support

For issues:
1. Check that RTL-SDR dongle is detected: `python test_sdr.py`
2. Verify drivers installed (Zadig with WinUSB)
3. Check USB connection quality
4. Try different FM frequencies
5. Check speaker/audio device settings

---

**Happy radio listening! 📻**
