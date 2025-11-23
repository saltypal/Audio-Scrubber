# GNU Radio OFDM with AI Denoising - Complete Workflow

## ✅ What You Have

1. **Working AI Model**: `saved_models/OFDM/ofdm_final_1dunet.pth`
   - Trained on GNU Radio OFDM waveforms
   - Ready to use immediately

2. **GNU Radio Installation**: v3.10.12.0
   - ✅ IIO support (for Adalm Pluto)
   - ❌ osmosdr (not needed)

3. **SDR Hardware**:
   - Adalm Pluto (TX) - works with IIO blocks
   - RTL-SDR (RX) - works with RTL-SDR Source block

4. **AI Integration Script**: `src/ofdm/utils/gnuradio_ai_block.py`
   - Ready to process IQ files
   - Can be integrated into GNU Radio flowgraphs

## 🎯 Complete Workflow

### Method A: File-Based Processing (Easiest)

**Step 1: Transmit with GNU Radio**
```
File Source → Stream to Tagged Stream → Stream CRC32 → OFDM TX → IIO Pluto Sink
```

**Step 2: Receive and Save**
```
RTL-SDR Source → Channel Model (Noise) → File Sink (rx_noisy.iq)
```

**Step 3: AI Denoising (Python)**
```powershell
python src/ofdm/utils/gnuradio_ai_block.py
```

**Step 4: Decode in GNU Radio**
```
File Source (rx_denoised.iq) → OFDM RX → Stream CRC32 → File Sink
```

### Method B: Real-Time Integration (Advanced)

Create custom GNU Radio Python block:
```
RTL-SDR Source → [AI Denoiser Block] → OFDM RX → Stream CRC32 → File Sink
```

## 📋 Quick Start Commands

### 1. Test with Existing Data
```powershell
# Your custom OFDM generated this file (but AI can't denoise it)
# Use GNU Radio to generate compatible data instead
```

### 2. Process IQ File with AI
```powershell
python -c "
from src.ofdm.utils.gnuradio_ai_block import process_iq_file
process_iq_file(
    'dataset/OFDM/rx_waveform_noisy.iq',
    'dataset/OFDM/rx_waveform_denoised.iq',
    'saved_models/OFDM/ofdm_final_1dunet.pth'
)
"
```

### 3. Check Results
```powershell
# Compare file sizes (should be identical)
Get-Item dataset/OFDM/rx_waveform_*.iq | Select Name, Length
```

## 🔧 GNU Radio Flowgraph Parameters

Match these exactly to your training data:

```python
OFDM Transmitter:
  - FFT Length: 64
  - Cyclic Prefix: 16
  - Occupied Carriers: ((-4, -3, -2, -1, 1, 2, 3, 4),)
  - Pilot Carriers: ((-21, -7, 7, 21),)
  - Pilot Symbols: ((1, 1, 1, -1),)
  - Header/Payload Modulation: QPSK

Stream to Tagged Stream:
  - Packet Length: 96
  - Length Tag Key: "packet_len"

Stream CRC32:
  - Generates Mode: Yes
  - Packet Length: 36 (metadata, not data size)
  - Length Tag Name: "packet_len"
```

## 🚀 Expected Results

- **Without AI**: Some packets lost at low SNR
- **With AI**: Improved packet recovery at low SNR
- **Metric**: Packet Error Rate (PER) should decrease

## 📁 File Structure

```
dataset/OFDM/
├── tx_waveform.iq          # From your Python code (incompatible with AI)
├── rx_waveform_noisy.iq    # From GNU Radio (compatible)
└── rx_waveform_denoised.iq # After AI processing

saved_models/OFDM/
└── ofdm_final_1dunet.pth   # Your trained model

src/ofdm/utils/
└── gnuradio_ai_block.py    # AI processing script
```

## ⚙️ Installation Check

```powershell
# Verify GNU Radio
python -c "from gnuradio import gr; print('GNU Radio:', gr.version())"

# Verify IIO (Pluto)
python -c "from gnuradio import iio; print('IIO: OK')"

# Verify AI Model
python -c "from src.ofdm.model.neuralnet import OFDM_UNet; import torch; print('PyTorch:', torch.__version__)"
```

## 🎓 Why This Works

Your AI model was trained on:
- GNU Radio's exact OFDM implementation
- Specific pilot structure and power levels
- GNU Radio's IFFT normalization

Your custom Python OFDM:
- Works perfectly (0% BER at 10dB SNR) ✅
- But uses different implementation details
- AI model can't recognize the waveform structure ❌

**Solution**: Use GNU Radio for TX/RX, use AI only for denoising the waveform.

## 📊 Next Steps

1. ✅ Create GNU Radio flowgraph matching your screenshot
2. ✅ Generate test transmission
3. ✅ Capture with RTL-SDR
4. ✅ Apply AI denoising
5. ✅ Decode and measure PER improvement

---

**Status**: Ready to deploy with GNU Radio! 🚀
