#!/usr/bin/env python3
"""
================================================================================
GNU RADIO AI DENOISING BLOCK
================================================================================
This script can be called from GNU Radio Companion (GRC) as an embedded Python
block or external script to apply AI denoising to OFDM waveforms.

Usage in GNU Radio:
    1. Use "Python Block" or "Embedded Python Block"
    2. Import this script
    3. Feed IQ samples → AI Denoise → Output clean IQ

Standalone Usage:
    python gnuradio_ai_block.py input.iq output.iq [model_path]
================================================================================
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Try to import GNU Radio
try:
    from gnuradio import gr
    GR_AVAILABLE = True
except ImportError:
    GR_AVAILABLE = False


# Model import
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from src.ofdm.model.neuralnet import OFDM_UNet
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("⚠️  Warning: OFDM_UNet model not found. AI denoising will be disabled.")


class AIDenoiser:
    """
    AI Denoising Engine for OFDM Waveforms
    Can be used standalone or embedded in GNU Radio
    """
    
    def __init__(self, model_path: str, chunk_size: int = 1024, device: str = 'auto'):
        """
        Args:
            model_path: Path to trained PyTorch model (.pth)
            chunk_size: Process in chunks (must match training)
            device: 'auto', 'cuda', or 'cpu'
        """
        self.chunk_size = chunk_size
        self.model = None
        self.device = None
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load PyTorch model"""
        if not MODEL_AVAILABLE:
            print("❌ Model class not available")
            return False
        
        path = Path(model_path)
        if not path.exists():
            print(f"❌ Model file not found: {model_path}")
            return False
        
        try:
            self.model = OFDM_UNet(in_channels=2, out_channels=2).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"✅ Model loaded: {path.name} on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
            return False
    
    def denoise(self, iq_samples: np.ndarray) -> np.ndarray:
        """
        Apply AI denoising to IQ waveform
        
        Args:
            iq_samples: Complex numpy array (np.complex64 or np.complex128)
            
        Returns:
            Denoised IQ samples (same length as input)
        """
        if self.model is None:
            print("⚠️  Model not loaded, returning original signal")
            return iq_samples
        
        # Ensure complex64
        iq_samples = iq_samples.astype(np.complex64)
        
        # Normalize input power (model expects normalized signals)
        input_power = np.mean(np.abs(iq_samples)**2)
        normalized = iq_samples / np.sqrt(input_power + 1e-10)
        
        # Pad to chunk size
        pad_len = (self.chunk_size - (len(normalized) % self.chunk_size)) % self.chunk_size
        if pad_len > 0:
            padded = np.pad(normalized, (0, pad_len), mode='constant')
        else:
            padded = normalized
        
        denoised_chunks = []
        
        with torch.no_grad():
            for i in range(0, len(padded), self.chunk_size):
                chunk = padded[i:i+self.chunk_size]
                
                # Prepare input: [Batch=1, Channels=2, Length]
                input_tensor = torch.stack([
                    torch.from_numpy(np.real(chunk)),
                    torch.from_numpy(np.imag(chunk))
                ]).unsqueeze(0).float().to(self.device)
                
                # Inference
                output_tensor = self.model(input_tensor)
                
                # Extract I/Q
                out_i = output_tensor[0, 0].cpu().numpy()
                out_q = output_tensor[0, 1].cpu().numpy()
                
                denoised_chunks.append(out_i + 1j*out_q)
        
        denoised = np.concatenate(denoised_chunks)[:len(normalized)]
        
        # Restore original power
        denoised_power = np.mean(np.abs(denoised)**2)
        denoised = denoised * np.sqrt(input_power / (denoised_power + 1e-10))
        
        return denoised.astype(np.complex64)


# ================================================================================
# GNU RADIO BLOCK (if gnuradio is available)
# ================================================================================

if GR_AVAILABLE:
    class ofdm_ai_denoiser(gr.sync_block):
        """
        GNU Radio sync block for AI denoising
        
        Usage in GRC:
            id: ofdm_ai_denoiser
            label: OFDM AI Denoiser
            category: AI Blocks
            
        Parameters:
            - model_path (string): Path to .pth model file
            - chunk_size (int): Processing chunk size (default 1024)
        """
        
        def __init__(self, model_path='saved_models/OFDM/unet1d_best.pth', chunk_size=1024):
            gr.sync_block.__init__(
                self,
                name='OFDM AI Denoiser',
                in_sig=[np.complex64],
                out_sig=[np.complex64]
            )
            
            self.denoiser = AIDenoiser(model_path, chunk_size)
            print(f"🧠 GNU Radio AI Denoiser Block Initialized")
        
        def work(self, input_items, output_items):
            """Process samples"""
            in0 = input_items[0]
            out0 = output_items[0]
            
            # Denoise
            denoised = self.denoiser.denoise(in0)
            
            # Output
            out0[:] = denoised
            
            return len(output_items[0])


# ================================================================================
# STANDALONE USAGE (Command Line)
# ================================================================================

def process_iq_file(input_path: str, output_path: str, model_path: str, chunk_size: int = 1024):
    """
    Process an IQ file through AI denoising
    
    Args:
        input_path: Input .iq file (complex64)
        output_path: Output .iq file (complex64)
        model_path: Path to model .pth
        chunk_size: Processing chunk size
    """
    print("="*60)
    print(" "*15 + "AI DENOISING (Standalone)")
    print("="*60)
    
    # Load IQ file
    print(f"\n📂 Loading: {input_path}")
    try:
        iq_data = np.fromfile(input_path, dtype=np.complex64)
        print(f"✅ Loaded {len(iq_data)} samples")
    except Exception as e:
        print(f"❌ Failed to load IQ file: {e}")
        return False
    
    # Initialize denoiser
    print(f"\n🧠 Initializing AI Denoiser...")
    denoiser = AIDenoiser(model_path, chunk_size)
    
    if denoiser.model is None:
        print("❌ Cannot proceed without model")
        return False
    
    # Denoise
    print(f"\n🔧 Applying AI Denoising...")
    denoised = denoiser.denoise(iq_data)
    print(f"✅ Denoising complete")
    
    # Save
    print(f"\n💾 Saving: {output_path}")
    try:
        denoised.tofile(output_path)
        print(f"✅ Saved {len(denoised)} samples")
    except Exception as e:
        print(f"❌ Failed to save: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ Processing Complete!")
    print("="*60)
    
    return True


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("="*60)
        print(" "*10 + "GNU RADIO AI DENOISING SCRIPT")
        print("="*60)
        print("\nUsage:")
        print("  python gnuradio_ai_block.py <input.iq> <output.iq> [model_path]")
        print("\nExample:")
        print("  python gnuradio_ai_block.py noisy.iq clean.iq saved_models/OFDM/unet1d_best.pth")
        print("\nDefault model: saved_models/OFDM/unet1d_best.pth")
        print("="*60)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_file = sys.argv[3] if len(sys.argv) > 3 else 'saved_models/OFDM/unet1d_best.pth'
    
    success = process_iq_file(input_file, output_file, model_file)
    
    sys.exit(0 if success else 1)
