# Why AI Model Needs Retraining - Detailed Explanation

## The Current Situation

### ✅ What Works
Your existing model (`saved_models/OFDM/ofdm_final_1dunet.pth`) **IS** trained on OFDM waveforms:
- Training data: `dataset/OFDM/clean_ofdm.iq` and `noisy_ofdm.iq`
- Format: Complex64 IQ samples (time-domain)
- Size: 163+ million samples
- Power: ~35.0 (matching our target)

### ❌ What's Wrong
The model fails on our new pipeline because of **MISMATCH between training data generation and our new implementation**:

## The Problem: Different OFDM Implementations

### Old Training Data (GNU Radio)
Your existing dataset was generated using **GNU Radio's OFDM implementation**:
```python
# Located in: src/ofdm/dataset_gnu.py
# Uses GNU Radio OFDM blocks
# Different parameters/structure than our new implementation
```

### New Pipeline (Our Implementation)
Our new code uses **custom OFDM implementation**:
```python
# Located in: src/ofdm/core/ofdm_pipeline.py
# FFT Size: 64
# CP Length: 16
# Data carriers: [-4, -3, -2, -1, 1, 2, 3, 4]
# Pilot carriers: [-21, -7, 7, 21]
```

## Key Differences

| Aspect | GNU Radio (Old) | New Pipeline |
|--------|----------------|--------------|
| OFDM Engine | GNU Radio blocks | Custom Python (NumPy FFT) |
| Subcarrier Mapping | GNU Radio default | Custom: 8 data + 4 pilots |
| Cyclic Prefix | Unknown (GNU Radio) | 16 samples |
| FFT Size | Unknown (GNU Radio) | 64 |
| Pilot Pattern | Unknown | BPSK: [1, 1, 1, -1] |
| Power Normalization | Unknown | Target: 35.0 |

## Why Retraining is Needed

The AI model learned to denoise **GNU Radio's specific OFDM structure**:
- Specific time-domain patterns from GNU Radio's IFFT
- Specific cyclic prefix structure
- Specific pilot locations and values
- Specific subcarrier arrangement

When we feed it our **custom OFDM structure**, it sees:
- Different time-domain patterns
- Different FFT artifacts
- Different pilot positions
- Different subcarrier mapping

**Result:** The model doesn't recognize the signal structure → produces garbage output → demodulation fails

## What to Retrain On

You have **TWO OPTIONS**:

### Option A: Retrain for New Pipeline (Recommended)

Generate new training data using our new implementation:

```python
from src.ofdm.core import OFDMTransceiver, add_awgn_noise
import numpy as np

transceiver = OFDMTransceiver()

# Generate clean/noisy pairs
clean_samples = []
noisy_samples = []

# Use diverse text data
training_texts = [
    "The quick brown fox jumps over the lazy dog",
    "Lorem ipsum dolor sit amet consectetur",
    # ... hundreds of different messages
]

for text in training_texts:
    # Generate clean OFDM waveform
    clean_waveform, _ = transceiver.transmit(text.encode('utf-8'))
    
    # Add noise at various SNR levels
    for snr_db in [0, 5, 10, 15, 20, 25]:
        noisy_waveform = add_awgn_noise(clean_waveform, snr_db)
        
        clean_samples.extend(clean_waveform)
        noisy_samples.extend(noisy_waveform)

# Save for training
np.array(clean_samples, dtype=np.complex64).tofile('dataset/OFDM/clean_ofdm_new.iq')
np.array(noisy_samples, dtype=np.complex64).tofile('dataset/OFDM/noisy_ofdm_new.iq')
```

**Then train:**
```bash
python src/ofdm/model/train_ofdm.py \
  --clean-data dataset/OFDM/clean_ofdm_new.iq \
  --noisy-data dataset/OFDM/noisy_ofdm_new.iq \
  --epochs 100
```

### Option B: Use GNU Radio Pipeline (Easier but Limited)

Keep the existing model and use GNU Radio for OFDM:
- Use GNU Radio Companion for TX/RX
- Call our AI denoising block from GNU Radio
- File: `src/ofdm/utils/gnuradio_ai_block.py`

**Pros:** Model already trained  
**Cons:** Locked into GNU Radio, less flexible

## The Actual Training Process

Your model is a **1D U-Net** that learns to map:
```
Input:  Noisy OFDM Waveform [I, Q channels, Length]
Output: Clean OFDM Waveform [I, Q channels, Length]
```

### What the AI Learns

1. **Time-Domain Structure Recognition**
   - OFDM symbols have specific time-domain patterns
   - Cyclic prefix creates correlation
   - FFT creates specific frequency content

2. **Noise vs Signal Separation**
   - Signal has structure (pilots, cyclic prefix, subcarrier arrangement)
   - Noise is random
   - AI learns to preserve structure, remove randomness

3. **Channel Effects**
   - Phase rotation
   - Amplitude scaling
   - Frequency-selective fading

### Training Data Requirements

For good performance, you need:
- **Diversity:** Many different messages (hundreds/thousands)
- **SNR Range:** Train on 0-25 dB SNR
- **Volume:** 100M+ samples (your current dataset has 163M ✅)
- **Balance:** Equal distribution across SNR levels

## Quick Fix: Generate Training Data

I can create a script to generate compatible training data:

```python
# File: src/ofdm/model/generate_training_data.py

from src.ofdm.core import OFDMTransceiver, add_awgn_noise
import numpy as np
from tqdm import tqdm

def generate_dataset(num_samples=100_000_000, output_dir='dataset/OFDM'):
    """
    Generate training dataset for OFDM AI denoising
    
    Args:
        num_samples: Target number of IQ samples (default: 100M)
        output_dir: Where to save clean_ofdm_new.iq and noisy_ofdm_new.iq
    """
    transceiver = OFDMTransceiver()
    
    # Training texts (use diverse data)
    texts = generate_diverse_texts()  # You'd implement this
    
    clean_data = []
    noisy_data = []
    
    snr_levels = [0, 5, 10, 15, 20, 25]
    
    with tqdm(total=num_samples) as pbar:
        while len(clean_data) < num_samples:
            for text in texts:
                # Transmit
                waveform, _ = transceiver.transmit(text.encode('utf-8'))
                
                # Multiple noise levels per message
                for snr in snr_levels:
                    noisy = add_awgn_noise(waveform, snr)
                    
                    clean_data.extend(waveform)
                    noisy_data.extend(noisy)
                    
                    pbar.update(len(waveform))
                    
                    if len(clean_data) >= num_samples:
                        break
                
                if len(clean_data) >= num_samples:
                    break
    
    # Save
    np.array(clean_data[:num_samples], dtype=np.complex64).tofile(
        f'{output_dir}/clean_ofdm_new.iq'
    )
    np.array(noisy_data[:num_samples], dtype=np.complex64).tofile(
        f'{output_dir}/noisy_ofdm_new.iq'
    )
```

## Bottom Line

**The model architecture is CORRECT** (1D U-Net for waveform denoising).

**The training approach is CORRECT** (learning clean vs noisy waveforms).

**The problem is:** Training data was generated with **different OFDM parameters** than our new implementation uses.

**The solution is:** Either:
1. Regenerate training data with new pipeline → Retrain (1-2 hours on GPU)
2. OR adapt our pipeline to match GNU Radio's parameters

**Recommended:** Option 1 - gives you full control and understanding of the system.

---

**Do you want me to:**
1. Create the training data generation script?
2. Modify the new pipeline to match GNU Radio parameters?
3. Create a hybrid approach using GNU Radio for OFDM + our AI?
