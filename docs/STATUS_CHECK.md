# AudioScrubber Status Check - Nov 17, 2025

## ✅ Issues Fixed

### 1. Unicode Encoding Errors in rtlsdr_denoise.py
**Problem:** Windows PowerShell couldn't encode emoji characters (✅, ❌, 🤖, 🔊, 📻, etc.)
**Solution:** Replaced all emojis with ASCII-safe text labels like `[SUCCESS]`, `[ERROR]`, `[AI]`, `[AUDIO]`, `[SDR]`
**Status:** ✅ FIXED - Script now runs without encoding errors

### 2. Missing soundfile Module in rtlfm_wrapper.py
**Problem:** `ModuleNotFoundError: No module named 'soundfile'`
**Solution:** Verified soundfile is installed (0.13.1) in AScrubber conda environment
**Status:** ✅ FIXED - Module is available, import now works

---

## ✅ AI Model Usage Confirmation

### YES - realtime_denoise IS using the saved AI model!

**Evidence:**

1. **Model Loading (Lines 65-90 in rtlsdr_denoise.py):**
   ```python
   def _load_model(self):
       """Loads the U-Net model from the specified path."""
       print(f"Loading model from '{self.model_path}'...")
       checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
       model.load_state_dict(checkpoint['model_state_dict'])
       print(f"[SUCCESS] Model loaded successfully!")
   ```
   - Default model path: `saved_models/unet1d_best.pth`
   - Model is loaded during initialization (constructor)
   - Runs `model.eval()` for inference mode

2. **Model Inference in AI Worker (Lines 190-230 in rtlsdr_denoise.py):**
   ```python
   def _ai_worker(self):
       # Buffers raw audio to match model's training size (44096 samples)
       input_buffer = np.array([], dtype=np.float32)
       model_frame_size = AudioSettings.AUDIO_LENGTH
       
       with torch.no_grad():
           # Reshape raw audio for model: [batch, channels, length]
           noisy_tensor = torch.from_numpy(frame_to_process).unsqueeze(0).unsqueeze(0).to(self.device)
           
           # DENOISE WITH U-NET MODEL
           clean_tensor = self.model(noisy_tensor)  # <-- AI MODEL INFERENCE HERE
           
           # Convert back to numpy for playback
           clean_chunk = clean_tensor.squeeze().cpu().numpy()
           self.processed_audio_queue.put(clean_chunk)
   ```

3. **3-Thread Pipeline:**
   - **SDR Thread:** Captures raw FM radio from RTL-SDR hardware
   - **AI Thread:** Runs denoising model on raw audio (LINE: `clean_tensor = self.model(noisy_tensor)`)
   - **Audio Thread:** Plays denoised output to speakers

4. **Bypass Option:**
   - Use `--no-ai` flag to skip AI processing and hear raw noisy signal
   - Confirms AI is only active when NOT bypassed

---

## 📊 Architecture Summary

```
RTL-SDR Hardware
       ↓
[SDR Thread] Captures FM radio, demodulates to audio
       ↓ (raw_audio_queue)
[AI Thread] RUNS SAVED MODEL for denoising
       ↓ (processed_audio_queue)
[Audio Thread] Plays denoised output to speakers
```

---

## 🚀 How to Test

### Test 1: Verify model loads
```bash
python src/rtlsdr_denoise.py --help
```
Should show usage without encoding errors.

### Test 2: Run with AI denoising (recommended)
```bash
python src/rtlsdr_denoise.py --frequency 99.5e6
```
Press Ctrl+C to stop. This will:
- Load the model from `saved_models/unet1d_best.pth`
- Capture FM radio at 99.5 MHz
- Run real-time denoising using the AI model
- Play denoised audio

### Test 3: Compare raw vs denoised
```bash
# Hear the NOISY original signal
python src/rtlsdr_denoise.py --frequency 99.5e6 --no-ai

# Hear the CLEAN denoised signal
python src/rtlsdr_denoise.py --frequency 99.5e6
```

---

## 📝 Next Steps

1. **Test Audio Output:** Run the denoiser and verify clean audio through speakers
2. **Record Noise Dataset:** Use `src/frequency_recorder.py` to build better training data
3. **Retrain Model:** Once you have real noise samples, retrain UNet1D for even better denoising
4. **Monitor Performance:** Use `src/fm_monitor.py` to find different stations and test across frequencies

---

## 🔧 Model Configuration

- **Model Type:** UNet1D (1D convolutional neural network)
- **Input Shape:** [batch_size, 1 channel, 44096 samples]
- **Output:** Denoised audio of same shape
- **Saved Location:** `saved_models/unet1d_best.pth`
- **Device:** Auto-selects CUDA (GPU) if available, otherwise CPU

---

Generated: 2025-11-17 | AudioScrubber Real-Time FM Denoiser
