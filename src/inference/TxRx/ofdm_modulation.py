"""
================================================================================
OFDM MODULATION CLASS - With AI Denoising Support
================================================================================
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from pathlib import Path
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ofdm.lib_archived.transceiver import OFDMTransmitter, OFDMReceiver
from ofdm.lib_archived.config import OFDMConfig
from ofdm.model.neuralnet import OFDM_UNet
from inference.TxRx.sdr_utils import SDRUtils


class OFDM_Modulation:
    """
    OFDM modulation/demodulation with optional AI denoising.
    """
    
    def __init__(self, use_ai=True, model_path=None, passthrough=False):
        """
        Initialize OFDM modulation.
        
        Args:
            use_ai: Enable AI denoising
            model_path: Path to trained model (if None, searches saved_models/OFDM/final_models)
            passthrough: Skip AI denoising (raw OFDM only)
        """
        self.use_ai = use_ai and not passthrough
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize OFDM transmitter and receiver
        self.config = OFDMConfig()
        self.transmitter = OFDMTransmitter(self.config)
        self.receiver = OFDMReceiver(self.config)
        
        # Load AI model if enabled
        if self.use_ai:
            self._load_model()
        
        print(f"🔧 OFDM Modulation initialized:")
        print(f"   AI Denoising: {self.use_ai}")
        print(f"   FFT Size: {self.config.fft_size}")
        print(f"   Cyclic Prefix: {self.config.cp_len}")
        print(f"   Data Carriers: {self.config.data_subcarriers_count}")
        if self.use_ai:
            print(f"   Model: {Path(self.model_path).name if self.model_path else 'None'}")
    
    def _load_model(self):
        """Load trained OFDM model."""
        # If no model path provided, search for best model
        if self.model_path is None:
            search_paths = [
                'saved_models/OFDM/final_models',
                'saved_models/OFDM',
                'x',  # Check x directory too
            ]
            
            for search_dir in search_paths:
                model_dir = Path(search_dir)
                if model_dir.exists():
                    # Look for 1D U-Net model first, then other candidates
                    candidates = list(model_dir.glob('*1dunet*.pth')) + \
                                list(model_dir.glob('unet1d*.pth')) + \
                                list(model_dir.glob('*best*.pth')) + \
                                list(model_dir.glob('ofdm_final*.pth'))
                    if candidates:
                        self.model_path = str(candidates[0])
                        break
        
        if self.model_path is None:
            print("⚠️  No model found. AI denoising disabled.")
            self.use_ai = False
            return
        
        try:
            self.model = OFDM_UNet(in_channels=2, out_channels=2)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle checkpoint dict vs direct state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded: {Path(self.model_path).name}")
        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            print(f"   AI denoising disabled.")
            self.use_ai = False
            self.model = None
    
    def modulate(self, data_bytes, image_path=None):
        """
        Modulate data to OFDM waveform.
        
        Args:
            data_bytes: Raw data bytes
            image_path: Optional path to original image for visualization
            
        Returns:
            Complex IQ waveform
        """
        print(f"🔄 Modulating {len(data_bytes)} bytes...")
        
        # Store original image for later comparison
        self.original_image = None
        if image_path and Path(image_path).suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            try:
                from PIL import Image
                self.original_image = Image.open(image_path)
                print(f"📷 Original image: {self.original_image.size}")
            except Exception as e:
                print(f"⚠️  Could not load image for visualization: {e}")
        
        # Transmit using OFDM transmitter
        waveform, info = self.transmitter.transmit(data_bytes)
        
        # Plot before TX
        model_name = Path(self.model_path).stem if self.model_path else "NoModel"
        SDRUtils.plot_waveform(
            waveform,
            "OFDM Modulated (Before TX)",
            f"BeforeTX_OFDM_{model_name}_waveform.png"
        )
        
        # Calculate actual power
        actual_power = np.mean(np.abs(waveform)**2)
        
        print(f"✅ Modulated to {len(waveform)} samples")
        print(f"   Symbols: {info.get('num_ofdm_symbols', 0)}")
        print(f"   TX Power: {actual_power:.3f} ({10*np.log10(actual_power + 1e-12):.2f} dB)")
        return waveform
    
    def demodulate(self, waveform):
        """
        Demodulate OFDM waveform with optional AI denoising.
        
        Args:
            waveform: Received complex IQ samples
            
        Returns:
            dict: {
                'data': decoded bytes (or None if failed),
                'control_data': decoded bytes without AI (for comparison),
                'stats': statistics dictionary,
                'waveform_noisy': noisy waveform,
                'waveform_denoised': denoised waveform (if AI used)
            }
        """
        print(f"🔄 Demodulating {len(waveform)} samples...")
        
        result = {
            'data': None,
            'control_data': None,
            'stats': {},
            'waveform_noisy': waveform,
            'waveform_denoised': None
        }
        
        # Control path (no AI)
        print("\n--- Control Path (No AI) ---")
        control_bytes, control_stats = self.receiver.receive(waveform)
        
        if control_bytes is not None and len(control_bytes) > 0:
            result['control_data'] = control_bytes
            print(f"✅ Control: Decoded {len(control_bytes)} bytes")
        else:
            print(f"❌ Control: Decoding failed")
        
        # AI denoising path
        if self.use_ai and self.model is not None:
            print("\n--- AI Denoising Path ---")
            
            # Denoise waveform
            denoised_waveform = self._denoise_waveform(waveform)
            result['waveform_denoised'] = denoised_waveform
            
            # Demodulate denoised waveform
            ai_bytes, ai_stats = self.receiver.receive(denoised_waveform)
            
            if ai_bytes is not None and len(ai_bytes) > 0:
                result['data'] = ai_bytes
                result['stats'] = ai_stats
                print(f"✅ AI Path: Decoded {len(ai_bytes)} bytes")
            else:
                print(f"❌ AI Path: Decoding failed")
                result['data'] = result['control_data']  # Fallback to control
                result['stats'] = control_stats
            
            # Plot comparison with images if available
            self._plot_denoising_results(
                waveform, denoised_waveform, control_stats, ai_stats,
                control_bytes, ai_bytes
            )
        else:
            # No AI, use control path result
            result['data'] = result['control_data']
            result['stats'] = control_stats
        
        return result
    
    def _denoise_waveform(self, waveform):
        """
        Apply AI denoising to waveform.
        
        Args:
            waveform: Noisy complex IQ samples
            
        Returns:
            Denoised complex IQ samples
        """
        # Prepare input (I/Q channels)
        waveform_2ch = np.stack([np.real(waveform), np.imag(waveform)], axis=0)
        waveform_tensor = torch.from_numpy(waveform_2ch).float().unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            denoised_tensor = self.model(waveform_tensor)
        
        # Convert back to complex
        denoised_2ch = denoised_tensor.cpu().numpy()[0]
        denoised_waveform = denoised_2ch[0] + 1j * denoised_2ch[1]
        
        print(f"🧠 AI Denoised: {len(denoised_waveform)} samples")
        return denoised_waveform
    
    def _plot_denoising_results(self, noisy, denoised, control_stats, ai_stats, 
                                  control_bytes=None, ai_bytes=None):
        """Plot comprehensive OFDM comparison with images and noise metrics."""
        from PIL import Image
        import io
        
        model_name = Path(self.model_path).stem if self.model_path else "NoModel"
        
        # Extract symbols for constellation
        noisy_symbols = self._extract_symbols(noisy)
        denoised_symbols = self._extract_symbols(denoised)
        
        # Try to reconstruct images from bytes
        original_img = self.original_image if hasattr(self, 'original_image') else None
        noisy_img = None
        clean_img = None
        
        if control_bytes:
            try:
                noisy_img = Image.open(io.BytesIO(control_bytes))
            except:
                pass
        
        if ai_bytes:
            try:
                clean_img = Image.open(io.BytesIO(ai_bytes))
            except:
                pass
        
        # Calculate noise metrics if we have images
        noise_metrics = {}
        if original_img and noisy_img:
            noise_metrics = self._calculate_noise_metrics(original_img, noisy_img, clean_img)
        
        output_dir = Path('src/inference/plot')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive comparison plot (3x2 grid)
        fig = plt.figure(figsize=(16, 18))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        # Row 1: Images (if available)
        if original_img or noisy_img or clean_img:
            # Original Image
            ax1 = fig.add_subplot(gs[0, 0])
            if original_img:
                ax1.imshow(original_img)
                ax1.set_title('📷 Original Image (Before TX)', fontweight='bold', fontsize=14)
            else:
                ax1.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=14)
                ax1.set_title('📷 Original Image', fontweight='bold', fontsize=14)
            ax1.axis('off')
            
            # Received Image (with noise metrics)
            ax2 = fig.add_subplot(gs[0, 1])
            if clean_img:
                ax2.imshow(clean_img)
                title = '📷 Received Image (After RX + AI)'
                if noise_metrics:
                    title += f"\nPSNR: {noise_metrics.get('psnr_clean', 0):.2f} dB"
                ax2.set_title(title, fontweight='bold', fontsize=14)
            elif noisy_img:
                ax2.imshow(noisy_img)
                title = '📷 Received Image (After RX, No AI)'
                if noise_metrics:
                    title += f"\nPSNR: {noise_metrics.get('psnr_noisy', 0):.2f} dB"
                ax2.set_title(title, fontweight='bold', fontsize=14)
            else:
                ax2.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=14)
                ax2.set_title('📷 Received Image', fontweight='bold', fontsize=14)
            ax2.axis('off')
        else:
            # No images - show noise metrics as text
            ax1 = fig.add_subplot(gs[0, :])
            ax1.text(0.5, 0.5, 'No image data available for visualization', 
                    ha='center', va='center', fontsize=14)
            ax1.axis('off')
        
        # QPSK reference points
        qpsk_ref = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        
        # Row 2: QPSK Constellations
        # Left: BEFORE AI Denoising (Noisy RX)
        ax3 = fig.add_subplot(gs[1, 0])
        if noisy_symbols is not None and len(noisy_symbols) > 0:
            ax3.scatter(np.real(noisy_symbols), np.imag(noisy_symbols),
                       alpha=0.4, s=15, c='orange', label='Noisy RX Symbols')
            ax3.scatter(np.real(qpsk_ref), np.imag(qpsk_ref),
                       c='red', s=250, marker='x', linewidths=4, label='Ideal QPSK', zorder=5)
        ax3.axhline(0, color='k', linewidth=0.8, alpha=0.3)
        ax3.axvline(0, color='k', linewidth=0.8, alpha=0.3)
        ax3.set_title('🔴 BEFORE AI Denoising - QPSK Constellation', fontweight='bold', fontsize=14)
        ax3.set_xlabel('I (In-Phase)', fontsize=11)
        ax3.set_ylabel('Q (Quadrature)', fontsize=11)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.axis('equal')
        ax3.set_xlim(-1.5, 1.5)
        ax3.set_ylim(-1.5, 1.5)
        
        # Right: AFTER AI Denoising (Clean)
        ax4 = fig.add_subplot(gs[1, 1])
        if denoised_symbols is not None and len(denoised_symbols) > 0:
            ax4.scatter(np.real(denoised_symbols), np.imag(denoised_symbols),
                       alpha=0.4, s=15, c='blue', label='AI Denoised Symbols')
            ax4.scatter(np.real(qpsk_ref), np.imag(qpsk_ref),
                       c='red', s=250, marker='x', linewidths=4, label='Ideal QPSK', zorder=5)
        ax4.axhline(0, color='k', linewidth=0.8, alpha=0.3)
        ax4.axvline(0, color='k', linewidth=0.8, alpha=0.3)
        ax4.set_title('🟢 AFTER AI Denoising - QPSK Constellation', fontweight='bold', fontsize=14)
        ax4.set_xlabel('I (In-Phase)', fontsize=11)
        ax4.set_ylabel('Q (Quadrature)', fontsize=11)
        ax4.legend(loc='upper right', fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.axis('equal')
        ax4.set_xlim(-1.5, 1.5)
        ax4.set_ylim(-1.5, 1.5)
        
        # Row 3: Waveform and PSD
        # Left: Waveform Comparison
        ax5 = fig.add_subplot(gs[2, 0])
        time_axis = np.arange(1000)
        ax5.plot(time_axis, np.real(noisy[:1000]), label='Noisy I', 
                alpha=0.6, linewidth=1.2, color='orange')
        ax5.plot(time_axis, np.real(denoised[:1000]), label='Denoised I', 
                alpha=0.8, linewidth=1.0, color='blue', linestyle='--')
        ax5.set_title('Waveform Comparison (I channel)', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Sample', fontsize=11)
        ax5.set_ylabel('Amplitude', fontsize=11)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Right: PSD Comparison
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.psd(noisy, NFFT=1024, Fs=2.0, scale_by_freq=False, 
               color='orange', alpha=0.7, label='Noisy')
        ax6.psd(denoised, NFFT=1024, Fs=2.0, scale_by_freq=False, 
               color='blue', alpha=0.7, label='Denoised')
        ax6.set_title('Power Spectral Density', fontweight='bold', fontsize=12)
        ax6.set_xlabel('Frequency (MHz)', fontsize=11)
        ax6.set_ylabel('Power (dB/Hz)', fontsize=11)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # Add stats as text (including noise metrics)
        stats_lines = [
            f"Control (No AI): BER={control_stats.get('ber', 0):.4f}, "
            f"Errors={control_stats.get('bit_errors', 0)}/{control_stats.get('total_bits', 0)}",
            f"AI Denoised:     BER={ai_stats.get('ber', 0):.4f}, "
            f"Errors={ai_stats.get('bit_errors', 0)}/{ai_stats.get('total_bits', 0)}"
        ]
        
        if noise_metrics:
            stats_lines.append("\n🔊 NOISE METRICS:")
            if 'snr_noisy' in noise_metrics:
                stats_lines.append(f"   Noisy RX:  SNR={noise_metrics['snr_noisy']:.2f} dB, "
                                 f"MSE={noise_metrics['mse_noisy']:.2f}, PSNR={noise_metrics['psnr_noisy']:.2f} dB")
            if 'snr_clean' in noise_metrics:
                stats_lines.append(f"   AI Clean:  SNR={noise_metrics['snr_clean']:.2f} dB, "
                                 f"MSE={noise_metrics['mse_clean']:.2f}, PSNR={noise_metrics['psnr_clean']:.2f} dB")
            if 'improvement' in noise_metrics:
                stats_lines.append(f"   🎯 Improvement: {noise_metrics['improvement']:.2f} dB SNR gain")
        
        stats_text = '\n'.join(stats_lines)
        fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='bottom')
        
        plt.suptitle(f'OFDM Transmission Analysis - Model: {model_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        filename = f'OFDM_Comparison_{model_name}.png'
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 OFDM comparison saved: {output_path}")
        
        # Print stats comparison
        print("\n📊 Statistics Comparison:")
        print(f"   Control: BER={control_stats.get('ber', 0):.4f}, "
              f"Errors={control_stats.get('bit_errors', 0)}/{control_stats.get('total_bits', 0)}")
        print(f"   AI:      BER={ai_stats.get('ber', 0):.4f}, "
              f"Errors={ai_stats.get('bit_errors', 0)}/{ai_stats.get('total_bits', 0)}")
        
        if noise_metrics:
            print("\n🔊 Noise Metrics:")
            if 'snr_noisy' in noise_metrics:
                print(f"   Noisy:  SNR={noise_metrics['snr_noisy']:.2f} dB, "
                      f"MSE={noise_metrics['mse_noisy']:.2f}, PSNR={noise_metrics['psnr_noisy']:.2f} dB")
            if 'snr_clean' in noise_metrics:
                print(f"   Clean:  SNR={noise_metrics['snr_clean']:.2f} dB, "
                      f"MSE={noise_metrics['mse_clean']:.2f}, PSNR={noise_metrics['psnr_clean']:.2f} dB")
            if 'improvement' in noise_metrics:
                print(f"   🎯 Improvement: {noise_metrics['improvement']:.2f} dB")
    
    def _calculate_noise_metrics(self, original_img, noisy_img, clean_img=None):
        """Calculate SNR, MSE, PSNR between images."""
        metrics = {}
        
        try:
            # Ensure same size
            if original_img.size != noisy_img.size:
                noisy_img = noisy_img.resize(original_img.size)
            if clean_img and clean_img.size != original_img.size:
                clean_img = clean_img.resize(original_img.size)
            
            # Convert to numpy arrays
            orig = np.array(original_img, dtype=np.float64)
            noisy = np.array(noisy_img, dtype=np.float64)
            
            # Calculate metrics for noisy image
            mse_noisy = np.mean((orig - noisy) ** 2)
            if mse_noisy > 0:
                psnr_noisy = 10 * np.log10(255.0 ** 2 / mse_noisy)
                signal_power = np.mean(orig ** 2)
                noise_power = mse_noisy
                snr_noisy = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
                
                metrics['mse_noisy'] = mse_noisy
                metrics['psnr_noisy'] = psnr_noisy
                metrics['snr_noisy'] = snr_noisy
            
            # Calculate metrics for clean image (if available)
            if clean_img:
                clean = np.array(clean_img, dtype=np.float64)
                mse_clean = np.mean((orig - clean) ** 2)
                if mse_clean > 0:
                    psnr_clean = 10 * np.log10(255.0 ** 2 / mse_clean)
                    noise_power_clean = mse_clean
                    snr_clean = 10 * np.log10(signal_power / noise_power_clean) if noise_power_clean > 0 else float('inf')
                    
                    metrics['mse_clean'] = mse_clean
                    metrics['psnr_clean'] = psnr_clean
                    metrics['snr_clean'] = snr_clean
                    
                    # Calculate improvement
                    if 'snr_noisy' in metrics:
                        metrics['improvement'] = snr_clean - snr_noisy
        
        except Exception as e:
            print(f"⚠️  Noise metrics calculation failed: {e}")
        
        return metrics
    
    def _extract_symbols(self, waveform):
        """Extract QPSK symbols from waveform for constellation plot.

        Uses config.data_carriers (not data_subcarriers) which holds the indices
        of active data subcarriers relative to DC.
        """
        try:
            # Remove CP and apply FFT
            symbol_size = self.config.fft_size + self.config.cp_len
            num_symbols = len(waveform) // symbol_size
            
            symbols_all = []
            for i in range(num_symbols):
                ofdm_symbol = waveform[i * symbol_size:(i + 1) * symbol_size]
                ofdm_symbol_no_cp = ofdm_symbol[self.config.cp_len:]
                freq_domain = np.fft.fft(ofdm_symbol_no_cp)
                
                # Extract data carriers (attribute is data_carriers)
                data_indices = self.config.data_carriers
                data_symbols = freq_domain[data_indices]
                symbols_all.extend(data_symbols)
            
            return np.array(symbols_all)
        except Exception as e:
            print(f"⚠️  Symbol extraction failed: {e}")
            return None
