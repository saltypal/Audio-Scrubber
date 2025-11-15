# Advanced Hyperparameter Tuning Guide

## Overview
The enhanced `backshot.py` now includes superior hyperparameter tuning capabilities with intelligent search, early stopping, and comprehensive result analysis.

## Key Improvements

### 1. Expanded Search Space
- **Batch sizes**: 8, 16, 32 (larger batches for stable gradients)
- **Learning rates**: 0.00005, 0.0001, 0.0002, 0.0005 (focused on effective range)
- **Epochs**: 75, 100, 150 (more training time for convergence)
- **Optimizers**: Adam and AdamW (AdamW has better regularization)
- **Scheduler patience**: 8, 12, 15 (more patience before reducing LR)
- **Scheduler factor**: 0.5, 0.3 (LR reduction amounts)
- **Early stopping patience**: 20, 25 (prevents premature stopping)

**Total combinations in FULL mode**: 3 × 4 × 3 × 2 × 3 × 2 × 2 = **864 combinations**

### 2. Quick Mode
For faster experimentation, use quick mode with reduced grid:
- **Batch sizes**: 8, 16
- **Learning rates**: 0.0001, 0.0005
- **Epochs**: 30, 50
- **Optimizers**: Adam only
- **Total combinations**: **8 combinations** (much faster!)

### 3. Advanced Training Features

#### Early Stopping
- Monitors validation loss and stops training if no improvement for N epochs
- Prevents overfitting and saves computation time
- Configurable patience parameter

#### Learning Rate Scheduling
- ReduceLROnPlateau: reduces learning rate when validation loss plateaus
- Configurable patience and reduction factor
- Minimum learning rate: 1e-7 to prevent too-small updates

#### Gradient Clipping
- Clips gradients to max norm of 1.0
- Prevents exploding gradients during training
- Improves training stability

#### Multiple Optimizer Support
- **Adam**: Standard adaptive learning rate optimizer
- **AdamW**: Adam with decoupled weight decay (often performs better)

### 4. Comprehensive Result Tracking
Each experiment records:
- Final train and validation losses
- Best validation loss achieved
- Number of epochs completed
- Train-validation gap (overfitting indicator)
- Model file path
- All hyperparameters used

### 5. Result Analysis
Built-in analysis function that shows:
- Top 5 best configurations
- Impact of each hyperparameter on performance
- Overfitting analysis
- Statistical summaries

## Usage

### Option 1: Quick Tuning (Recommended for testing)
```powershell
python src/model/backshot.py quick
```
- Tests 8 combinations
- Takes ~2-4 hours
- Good for initial exploration

### Option 2: Full Tuning (Best results)
```powershell
python src/model/backshot.py full
```
- Tests 864 combinations
- Takes ~2-7 days depending on hardware
- Finds optimal hyperparameters

### Option 3: Analyze Results
```powershell
python src/model/backshot.py analyze
```
- Displays top configurations
- Shows hyperparameter impact
- No training, just analysis

### Option 4: Default Training
```powershell
python src/model/backshot.py
```
- Trains with default config from Config class
- No hyperparameter search
- Fastest option

## Understanding the Results

### Results File Location
`saved_models/tuning/hyperparameter_results.json`

### Key Metrics

1. **Best Validation Loss**: Lower is better
   - Primary metric for model selection
   - Measures denoising quality on unseen data

2. **Train-Val Gap**: Smaller is better
   - Large gap indicates overfitting
   - Model memorizing training data rather than learning patterns
   - Target: < 0.01 for good generalization

3. **Epochs Completed**:
   - If < max epochs, early stopping was triggered
   - Indicates model converged early
   - Saves training time

### Model Files
Each experiment saves to:
```
saved_models/tuning/unet1d_bs{batch}_lr{lr}_ep{epochs}_{optimizer}_sp{patience}.pth
```

Best overall model is saved to:
```
saved_models/unet1d_best.pth
```

## What Makes This "Superior"?

### Before (Old System)
- ❌ Limited search space (18 combinations)
- ❌ Only tested batch size, LR, epochs
- ❌ No early stopping (wasted computation)
- ❌ Single optimizer type
- ❌ Fixed scheduler settings
- ❌ No overfitting detection

### After (New System)
- ✅ Massive search space (864 combinations)
- ✅ Tests optimizers, schedulers, stopping criteria
- ✅ Early stopping saves days of training time
- ✅ Multiple optimizer support (Adam, AdamW)
- ✅ Configurable scheduler behavior
- ✅ Tracks overfitting via train-val gap
- ✅ Gradient clipping for stability
- ✅ Quick mode for fast experimentation
- ✅ Built-in result analysis
- ✅ Comprehensive metrics tracking

## Tips for Best Results

1. **Start with Quick Mode**
   - Test the system and get initial results
   - Verify dataset and training work correctly
   - Takes only a few hours

2. **Then Run Full Mode**
   - Let it run overnight or over a weekend
   - Will find optimal configuration
   - Worth the wait for best quality

3. **Monitor Training**
   - Check that validation loss is decreasing
   - Watch for early stopping triggers
   - Look at train-val gap for overfitting

4. **Analyze Results**
   - Use `analyze` mode to understand patterns
   - See which hyperparameters matter most
   - Use insights for future experiments

5. **Hardware Considerations**
   - GPU recommended (much faster)
   - CPU training will take 10-50x longer
   - Consider cloud GPUs for full tuning

## Technical Details

### Early Stopping Logic
```python
if validation_loss < best_validation_loss:
    best_validation_loss = validation_loss
    epochs_no_improve = 0
    save_model()
else:
    epochs_no_improve += 1
    if epochs_no_improve >= patience:
        stop_training()
```

### Learning Rate Scheduling
- **Mode**: min (reduce when loss plateaus)
- **Factor**: 0.5 or 0.3 (multiply LR by this)
- **Patience**: 8-15 epochs (wait before reduction)
- **Min LR**: 1e-7 (don't go below this)

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Prevents gradients from exceeding norm of 1.0

## Expected Outcomes

### Good Results
- Validation loss: 0.001 - 0.01
- Train-val gap: < 0.01
- Early stopping after 30-100 epochs
- Clear audio denoising

### Signs of Problems
- Validation loss > 0.1 (poor denoising)
- Train-val gap > 0.05 (overfitting)
- No early stopping (not converging)
- High learning rate needed (data issues)

## Next Steps After Tuning

1. Test the best model on real audio
2. Use `src/inference.py` for batch denoising
3. Try `src/rtlsdr_denoise.py` for real-time RTL-SDR
4. Fine-tune with more training data if needed
5. Experiment with model architecture changes

---

**Remember**: The best hyperparameters depend on your specific dataset. What works for one dataset may not work for another. This system helps you find what works best for YOUR data.
