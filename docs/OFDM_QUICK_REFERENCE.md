# OFDM System - Quick Reference

## ✅ What We Fixed

**The Problem:** We were applying AI to QPSK symbols (WRONG!)  
**The Solution:** AI now processes OFDM waveforms (CORRECT!)

## 📁 New File Structure

```
src/ofdm/
├── core/                    # ✅ Use this for new work
│   ├── ofdm_pipeline.py
│   └── test_ai_denoising.py
├── utils/
│   ├── format_module.py     # Fix corrupted files
│   └── gnuradio_ai_block.py # GNU Radio integration
└── lib_archived/            # ⚠️ Old code (don't use)
```

## 🚀 Quick Commands

```bash
# Test the system
python src/ofdm/core/test_ai_denoising.py

# Denoise IQ file
python src/ofdm/utils/gnuradio_ai_block.py noisy.iq clean.iq

# Fix corrupted image
python src/ofdm/utils/format_module.py received.png
```

## 💻 Code Examples

### Transmit
```python
from src.ofdm.core import OFDMTransceiver

tx = OFDMTransceiver()
waveform, meta = tx.transmit(b"Hello!")
# Send waveform to SDR
```

### Receive with AI
```python
from src.ofdm.core import OFDMTransceiver
from src.ofdm.utils.gnuradio_ai_block import AIDenoiser

# Receive from SDR
noisy_waveform = sdr.receive()

# Denoise
denoiser = AIDenoiser('model.pth')
clean_waveform = denoiser.denoise(noisy_waveform)

# Decode
rx = OFDMTransceiver()
message, meta = rx.receive(clean_waveform)
print(message.decode('utf-8'))
```

## 📊 Current Status

| Component | Status |
|-----------|--------|
| OFDM TX/RX | ✅ Working |
| QPSK Modulation | ✅ Working |
| Channel Equalization | ✅ Working |
| File Format Fixer | ✅ Working |
| GNU Radio Block | ✅ Working |
| AI Model | ⚠️ Needs Retraining |

## ⚠️ Known Issue

The AI model needs retraining on OFDM waveforms.  
Control path (no AI) works perfectly - 0% BER at 10dB SNR.

## 📚 Documentation

- **Complete Guide:** `docs/OFDM_AI_PIPELINE.md`
- **Summary:** `docs/OFDM_IMPLEMENTATION_SUMMARY.md`
- **This File:** Quick reference

## 🔧 Troubleshooting

**Q: AI path fails?**  
A: Model needs retraining on OFDM waveforms

**Q: File corrupted after RX?**  
A: Use `format_module.py` to fix headers

**Q: How to use in GNU Radio?**  
A: See `gnuradio_ai_block.py` for block and examples

---

**Last Updated:** November 23, 2025
