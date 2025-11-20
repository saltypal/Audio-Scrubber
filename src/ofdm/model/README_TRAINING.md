# OFDM Denoising Model Training

Complete pipeline for training a 1D U-Net to denoise OFDM/QPSK signals.

## 📁 Directory Structure

```
src/ofdm/model/
├── neuralnet.py          # OFDM_UNet architecture
├── train_ofdm.py         # Main training script
├── inspect_dataset.py    # Dataset inspection utility
├── test_model.py         # Model testing script
└── backshot_ofdm.py      # Legacy GNU Radio training (for reference)

dataset/OFDM/
├── clean_ofdm.iq         # Clean IQ samples (1.17 GB)
└── noisy_ofdm.iq         # Noisy IQ samples (1.17 GB)

saved_models/OFDM/
├── unet1d_best.pth       # Best model (by validation loss)
├── unet1d_final.pth      # Final model after training
├── unet1d_epoch_*.pth    # Checkpoints every 10 epochs
└── training_curve.png    # Loss visualization
```

## 🚀 Quick Start

### 1. Inspect Dataset (Optional)
```bash
python src/ofdm/model/inspect_dataset.py
```

**Output:**
- File sizes and sample counts
- Data ranges and types
- Estimated SNR
- Training batch calculations

### 2. Train Model
```bash
# Basic training (50 epochs, batch size 32)
python src/ofdm/model/train_ofdm.py

# Custom parameters
python src/ofdm/model/train_ofdm.py --epochs 100 --batch_size 64 --lr 0.0005

# GPU training (auto-detected)
python src/ofdm/model/train_ofdm.py --device cuda
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--chunk_size`: Samples per chunk (default: 1024)
- `--val_split`: Validation split ratio (default: 0.1)
- `--device`: Device (cuda/cpu/auto, default: auto)

### 3. Test Model
```bash
python src/ofdm/model/test_model.py
```

**Output:**
- SNR improvement metrics
- Constellation diagrams (clean, noisy, denoised)
- Time-domain comparison plots
- Saved visualization: `saved_models/OFDM/test_results.png`

## 📊 Dataset Details

**Format:** Raw complex64 IQ samples  
**Clean File:** 1.17 GB (~146M samples)  
**Noisy File:** 1.17 GB (~146M samples)  

**Generation Parameters:**
- **Sample Rate:** 2 MHz
- **FFT Size:** 64 subcarriers
- **Cyclic Prefix:** 16 samples
- **Modulation:** QPSK (4-QAM)
- **SNR Range:** -5 to 30 dB (wide range for robustness)
- **Frequency Offset:** ±40 kHz (handles RTL-SDR drift)

**Training Chunks:**
- Chunk Size: 1024 samples
- Total Chunks: ~142,000
- Train/Val Split: 90/10 (~128k / ~14k)

## 🧠 Model Architecture

**OFDM_UNet** - 1D U-Net for IQ Denoising

```
Input:  (Batch, 2, 1024) → [I, Q] channels
Output: (Batch, 2, 1024) → [I, Q] cleaned

Encoder:  32 → 64 → 128 → 256 → 512 (bottleneck)
Decoder:  512 → 256 → 128 → 64 → 32 → 2 (output)

Parameters: ~1.2M trainable parameters
Loss: MSE (Mean Squared Error)
Optimizer: Adam with ReduceLROnPlateau scheduler
```

## 📈 Training Output

Example training session:
```
======================================================================
                OFDM DENOISING MODEL TRAINER
======================================================================
🖥️  Device: cuda
📦 Batch Size: 32
📚 Epochs: 50
📊 Learning Rate: 0.001
======================================================================

📁 Loading dataset...
✅ Loaded 146,210,560 samples
   Total chunks: 142,783

📊 Dataset Split:
   Training: 128,504 chunks
   Validation: 14,279 chunks

🧠 Model: OFDM_UNet
   Parameters: 1,237,058

======================================================================
🚀 Starting Training...
======================================================================

📍 Epoch 1/50
Training: 100%|████████| 4016/4016 [02:15<00:00]
Validation: 100%|████████| 447/447 [00:08<00:00]
   Train Loss: 0.008234
   Val Loss:   0.007891
   ✅ Best model saved! (Val Loss: 0.007891)

...

📍 Epoch 50/50
   Train Loss: 0.000542
   Val Loss:   0.000598
   ✅ Best model saved! (Val Loss: 0.000598)

======================================================================
✅ TRAINING COMPLETE
======================================================================
📁 Models saved in: saved_models\OFDM
🏆 Best validation loss: 0.000598
📈 Total epochs: 50
======================================================================
```

## 🎯 Performance Expectations

**Typical Results:**
- **Input SNR:** 10 dB (noisy)
- **Output SNR:** 20-25 dB (after denoising)
- **SNR Improvement:** 10-15 dB
- **Constellation Cleanup:** Tight clustering around QPSK points
- **Training Time:** ~2-3 hours (GPU), ~12-15 hours (CPU)

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size
```bash
python src/ofdm/model/train_ofdm.py --batch_size 16
```

### Issue: Dataset not found
**Solution:** Generate dataset first
```bash
python dataset_ofdm/ofdm_dataset_creation.py
```

### Issue: Slow training
**Solution:** Enable GPU if available
- Check GPU: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

## 📝 Usage in SDR System

After training, the model is automatically used by:
- `src/ofdm/TxRx/USE_OFDM.py` (with `--denoise` flag)
- `src/ofdm/TxRx/sdr_hardware.py` (RTLSDRReceiver class)

Example:
```bash
# Receive with denoising
python src/ofdm/TxRx/USE_OFDM.py --mode rx --denoise

# Loopback test with denoising
python src/ofdm/TxRx/USE_OFDM.py --mode loopback --type file --file test.png --denoise
```

## 🔬 Advanced: Custom Training

Modify `train_ofdm.py` for experiments:
- **Data Augmentation:** Add random phase shifts, amplitude scaling
- **Loss Functions:** Try L1 loss, perceptual loss
- **Architecture:** Adjust U-Net depth, channel counts
- **Regularization:** Add dropout, weight decay

---

**Last Updated:** November 20, 2025  
**Status:** Production Ready ✅
