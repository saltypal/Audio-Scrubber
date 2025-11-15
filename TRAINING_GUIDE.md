# Training Guide for Audio Denoiser

## 🎯 8-Hour Training Optimization

### Current Settings (Optimized for 8 hours)
- **Total Epochs**: 40 (~12 min/epoch = 480 min = 8 hours)
- **Batch Size**: 16 (good GPU utilization)
- **Learning Rate**: 0.0001 (proven for audio)
- **Scheduler Patience**: 3 epochs (~36 min before LR reduction)
- **Early Stopping**: 6 epochs (~72 min of no improvement)

### Training Timeline
```
Hour 0-1:   Epochs 1-5   (Initial training, high loss)
Hour 1-3:   Epochs 6-15  (Rapid improvement)
Hour 3-5:   Epochs 16-25 (Convergence phase)
Hour 5-7:   Epochs 26-35 (Fine-tuning, possible early stop)
Hour 7-8:   Epochs 36-40 (Final refinement or stopped early)
```

### Resume Training (Critical Feature!)

**Start training:**
```bash
python src/model/backshot.py
```

**Resume if interrupted:**
```bash
python src/model/backshot.py resume
```

**Resume from specific checkpoint:**
```bash
python src/model/backshot.py resume saved_models/checkpoints/latest_checkpoint.pth
```

**What gets saved in checkpoints:**
- Model weights
- Optimizer state
- Learning rate scheduler state
- Training/validation loss history
- Epoch number
- Best validation loss
- Early stopping counter

## 📊 Dataset Questions Answered

### Question 1: Should we change the noise addition method?

**Current Method:**
```python
# We add 5 noise levels per clean file
NOISE_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.1]
# Result: 5567 clean files → 27,835 noisy files (5x multiplier)
```

**Recommendation: Keep it as is! Here's why:**

✅ **Advantages of current approach:**
1. **Data Augmentation**: 5x more training samples from same audio
2. **Robustness**: Model learns to handle varying noise levels
3. **Real-world variety**: Speech has different SNR levels
4. **No storage overhead**: Generated on-the-fly or pre-computed once

❌ **Alternative (not recommended):**
- Single noise level: Only 5,567 samples (insufficient for deep learning)
- Random noise per epoch: Non-reproducible results
- More noise levels: Diminishing returns after 5 levels

**Dataset Size Impact:**
| Method | Clean Files | Noisy Files | Training Time |
|--------|-------------|-------------|---------------|
| Current (5 levels) | 5,567 | 27,835 | ~12 min/epoch |
| Single level | 5,567 | 5,567 | ~3 min/epoch ⚠️ (underfits!) |
| 10 levels | 5,567 | 55,670 | ~24 min/epoch ⚠️ (exceeds 8hr) |

**Verdict:** Current method is optimal for 8-hour window!

### Question 2: Does dataset size matter?

**Yes, critically!** Deep learning rule of thumb:
- **Minimum**: 10,000 samples (bare minimum)
- **Good**: 20,000-50,000 samples ✅ **(You have 27,835 - perfect!)**
- **Excellent**: 100,000+ samples

**Your dataset (27,835 samples):**
- Split: 22,268 training + 5,567 validation
- Batches per epoch: 22,268 / 16 = 1,392 batches
- **This is ideal** for U-Net architecture!

### Question 3: Alternative Noise Addition Methods

**Option A: Dynamic Noise Mixing (Advanced)**
```python
# Mix multiple noise types
noise_types = ['white', 'pink', 'brown', 'ambient']
# Pros: More realistic
# Cons: Complex, needs noise sample library
```

**Option B: SNR-based Noise (Engineering approach)**
```python
# Add noise to specific Signal-to-Noise Ratio
target_SNR = [0, 5, 10, 15, 20] # dB
# Pros: Industry standard, measurable
# Cons: Requires SNR calculation
```

**Option C: Keep it simple (Recommended for now)**
```python
# Current: amplitude-based noise levels
NOISE_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.1]
# Pros: Simple, reproducible, works well
# Cons: Not true SNR measurement
```

**Recommendation:** Start with current method, optimize later if needed.

## 🎵 Training on Music Dataset (Question 3)

### Important Considerations

**1. Different Acoustic Properties:**
```
Speech:
- Harmonic structure: Pitch + formants
- Temporal: Short bursts (phonemes)
- Frequency range: 80-8000 Hz (mainly)
- Sparsity: High (lots of silence)

Music:
- Harmonic structure: Multiple instruments
- Temporal: Sustained notes, polyphony
- Frequency range: 20-20000 Hz (full spectrum)
- Sparsity: Low (dense audio)
```

**2. Model Architecture Implications:**
- ✅ Same U-Net architecture works for both
- ⚠️ May need different `AUDIO_LENGTH` (music has longer context)
- ⚠️ May need higher sample rate (44100 Hz instead of 22050 Hz)

**3. Training Strategy for Music:**

**Option A: Separate Model (Recommended)**
```bash
# Train speech model first
python src/model/backshot.py

# Then train music model with different config
# Create config_music.py with:
# - SAMPLE_RATE = 44100 Hz (full quality)
# - AUDIO_LENGTH = 88192 (2 seconds @ 44.1kHz, divisible by 16)
# - Different dataset paths
```

**Option B: Transfer Learning**
```python
# Train on speech first (faster convergence)
# Then fine-tune on music dataset
# Pros: Leverages learned features
# Cons: Risk of speech bias
```

**Option C: Multi-domain Model**
```python
# Mix speech + music in single dataset
# Pros: One model for all
# Cons: May not excel at either
```

### Recommended Workflow for Music:

1. **Prepare Music Dataset:**
   ```bash
   # Create music-specific processor
   src/dataset_creation/process_music_dataset.py
   ```

2. **Update Config:**
   ```python
   # config_music.py
   SAMPLE_RATE = 44100  # Higher quality for music
   AUDIO_LENGTH = 88192  # 2 seconds, divisible by 16
   DATASET_PATH = "dataset/music_processed"
   ```

3. **Train Separately:**
   ```bash
   # Speech model (8 hours)
   python src/model/backshot.py
   
   # Music model (another 8 hours)
   python src/model/backshot.py --config config_music.py
   ```

4. **Compare Results:**
   - Speech denoising: Use speech test set
   - Music denoising: Use music test set
   - Cross-domain: Test speech model on music (usually fails)

## 🚀 Quick Start Commands

### Standard Training (8 hours)
```bash
python src/model/backshot.py
```

### Resume Training
```bash
# If you need to stop and continue later
python src/model/backshot.py resume
```

### Check Progress
```bash
# View checkpoints
ls saved_models/checkpoints/

# Analyze training (after completion)
python src/model/backshot.py analyze
```

## 📈 Expected Results

**After 8 hours of training:**
- Best validation loss: ~0.002-0.005 (MSE)
- Training loss: ~0.001-0.003 (MSE)
- Model size: ~50MB
- Checkpoints: Saved every epoch
- Final plot: `saved_models/training_history.png`

## ⚠️ Important Notes

1. **Don't change noise method now** - current approach is optimal
2. **Dataset size (27,835) is perfect** - don't worry about it
3. **Music needs separate model** - different acoustic properties
4. **Always test resume** - run for 1 epoch, stop, resume to verify
5. **Monitor first 5 epochs** - should see steady loss decrease

## 🎯 Success Criteria

✅ Training completes in 8 hours or early stops
✅ Can resume from any checkpoint
✅ Validation loss decreases steadily
✅ No overfitting (train-val gap < 0.001)
✅ Plot shows convergence curve

Happy training! 🎉
