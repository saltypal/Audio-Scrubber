"""
MAIN INFERENCE SCRIPT - Adalm Pluto TX + RTL-SDR RX with AI Denoising (FM only)

Usage:
    python main_inference.py --audio <audio_file> [--fm-mode <mode>] [--fm-architecture <arch>]
    python main_inference.py --passthrough --audio <audio_file>
"""

import sys
import time
import numpy as np
import argparse
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.TxRx.sdr_base import PlutoSDR, PlutoRX, RTLSDR, SDR_CONFIG, FM_CONFIG
from inference.TxRx.fm_modulation import FM_Modulation
from inference.TxRx.sdr_utils import SDRUtils


def main():
    parser = argparse.ArgumentParser(description='SDR TX/RX with AI Denoising (FM)')
    parser.add_argument('--audio', type=str, required=True, help='Audio file to transmit')
    parser.add_argument('--model', type=str, help='Path to model (optional, auto-detected)')
    parser.add_argument('--passthrough', action='store_true',
                       help='Disable AI denoising (raw modulation only)')
    parser.add_argument('--rx-duration', type=float, default=5.0,
                       help='RX capture duration in seconds (default: 5.0)')
    parser.add_argument('--freq', type=float, help='Center frequency in MHz (default: 105)')
    parser.add_argument('--tx-gain', type=int, help='TX gain in dB (default: -10)')
    parser.add_argument('--output', type=str, help='Output file for received audio')
    
    # FM specific
    parser.add_argument('--fm-mode', type=str, default='general',
                       choices=['general', 'music', 'speech'],
                       help='FM model type (default: general)')
    parser.add_argument('--fm-architecture', type=str, default=None,
                       choices=['1dunet', 'stft'],
                       help='FM model architecture: 1dunet or stft (default: auto-detect)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FM SDR INFERENCE WITH AI DENOISING")
    print("="*80)
    print(f"AI Denoising: {'Disabled (Passthrough)' if args.passthrough else 'Enabled'}")
    print("="*80)
    
    # ========== STEP 1: Initialize Hardware ==========
    print("\n📡 STEP 1: Initialize Hardware")
    print("-" * 80)
    
    pluto = PlutoSDR()
    rx_sdr = RTLSDR()
    rx_device_name = "RTL-SDR RX"
    
    if not pluto.check_device():
        print("❌ TX Pluto not available. Aborting.")
        return
    
    if not rx_sdr.check_device():
        print(f"❌ {rx_device_name} not available. Aborting.")
        return
    
    # Configure hardware for FM
    config = FM_CONFIG
    freq = (args.freq * 1e6) if args.freq else config['center_freq']
    tx_gain = args.tx_gain if args.tx_gain is not None else config['tx_gain']
    
    pluto.configure(freq=freq, gain=tx_gain)
    rx_sdr.configure(freq=freq)
    
    print(f"📻 Frequency: {freq/1e6:.3f} MHz (FM mode)")
    print(f"📡 TX Gain: {tx_gain} dB")
    print(f"📥 RX Device: {rx_device_name}")
    
    # ========== STEP 2: Initialize Modulation ==========
    print("\n🔧 STEP 2: Initialize FM Modulation")
    print("-" * 80)
    
    modulation = FM_Modulation(
        use_ai=not args.passthrough,
        model_path=args.model,
        passthrough=args.passthrough,
        mode=args.fm_mode,
        architecture=args.fm_architecture
    )
    data_path = args.audio
    
    # ========== STEP 3: Load and Modulate Audio ==========
    print("\n📦 STEP 3: Load and Modulate Audio")
    print("-" * 80)
    
    # Check if real-time streaming mode (file input)
    if isinstance(data_path, str) and Path(data_path).exists():
        print("🎵 Using REAL-TIME streaming mode for FM")
        # Get generator for streaming
        tx_stream = modulation.modulate(data_path, realtime=True, chunk_size=44100)
        tx_waveform = None  # Will be streamed
    else:
        # Load audio and modulate
        tx_waveform = modulation.modulate(data_path)
        
        # Check if modulation failed
        if tx_waveform is None:
            print("❌ Modulation failed. Check audio file path.")
            pluto.stop()
            rx_sdr.stop()
            return
    
    # ========== STEP 4: Transmit ==========
    print("\n📤 STEP 4: Start Transmission")
    print("-" * 80)
    
    # Check if FM real-time streaming
    if tx_waveform is None:
        # Real-time streaming mode - continuous TX and RX with playback
        print("🔴 LIVE FM TRANSMISSION (Real-time with Audio Playback)")
        print("   Press Ctrl+C to stop...")
        
        import sounddevice as sd
        import matplotlib.pyplot as plt
        
        # Audio buffer for playback
        audio_buffer = []
        
        # Setup live plotting (same layout as live_denoise.py)
        plt.ion()  # Interactive mode
        fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('🔴 LIVE FM Transmission & Denoising', fontsize=14, fontweight='bold')
        
        # Waveform plot
        x = np.arange(modulation.audio_rate)  # 1 second of samples
        line_noisy, = ax_wave.plot(x, np.zeros(modulation.audio_rate), color='orange', alpha=0.6, label='Noisy (After RX)')
        line_clean, = ax_wave.plot(x, np.zeros(modulation.audio_rate), color='blue', alpha=0.8, label='Denoised (AI)')
        ax_wave.set_xlim(0, modulation.audio_rate)
        ax_wave.set_ylim(-1, 1)
        ax_wave.set_xlabel('Sample')
        ax_wave.set_ylabel('Amplitude')
        ax_wave.set_title('Audio Waveform Comparison (1 second window)')
        ax_wave.legend(loc='upper right')
        ax_wave.grid(True, alpha=0.3)
        
        # Spectrum plot
        freqs = np.fft.rfftfreq(2048, 1/modulation.audio_rate)
        line_spec_noisy, = ax_spec.plot(freqs, np.zeros_like(freqs), color='orange', alpha=0.6, label='Noisy')
        line_spec_clean, = ax_spec.plot(freqs, np.zeros_like(freqs), color='blue', alpha=0.8, label='Denoised')
        ax_spec.set_xlim(0, modulation.audio_rate // 2)
        ax_spec.set_ylim(-80, 10)
        ax_spec.set_xlabel('Frequency (Hz)')
        ax_spec.set_ylabel('Magnitude (dB)')
        ax_spec.set_title('Frequency Spectrum')
        ax_spec.legend(loc='upper right')
        ax_spec.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
        
        try:
            chunk_num = 0
            for fm_chunk in tx_stream:
                chunk_num += 1
                
                if chunk_num == 1:
                    # Start transmission with first chunk
                    pluto.transmit_background(fm_chunk)
                    print(f"   📡 Chunk {chunk_num}: Started transmission")
                    time.sleep(0.3)  # Reduced delay
                
                # Receive while transmitting
                rx_duration = len(fm_chunk) / config['sample_rate']
                rx_chunk = rx_sdr.receive(duration=rx_duration)
                
                if rx_chunk is not None:
                    # Demodulate chunk
                    result = modulation.demodulate(rx_chunk)
                    
                    if result['audio'] is not None:
                        noisy_audio = result['control_audio']
                        clean_audio = result['audio']
                        
                        # Play audio immediately (non-blocking)
                        print(f"   🔊 Chunk {chunk_num}: Playing {len(clean_audio)/modulation.audio_rate:.2f}s")
                        sd.play(clean_audio, modulation.audio_rate, blocking=False)
                        audio_buffer.append(clean_audio)
                        
                        # Update plots (fast update - show last 1 second)
                        if chunk_num % 1 == 0:  # Update every chunk for smooth animation
                            # Pad or trim to exactly 1 second
                            display_len = modulation.audio_rate
                            if len(noisy_audio) >= display_len:
                                noisy_display = noisy_audio[:display_len]
                                clean_display = clean_audio[:display_len]
                            else:
                                noisy_display = np.pad(noisy_audio, (0, display_len - len(noisy_audio)))
                                clean_display = np.pad(clean_audio, (0, display_len - len(clean_audio)))
                            
                            # Update waveforms
                            line_noisy.set_ydata(noisy_display)
                            line_clean.set_ydata(clean_display)
                            
                            # Update spectrum (FFT)
                            noisy_fft = np.fft.rfft(noisy_display, n=2048)
                            clean_fft = np.fft.rfft(clean_display, n=2048)
                            
                            noisy_mag_db = 20 * np.log10(np.abs(noisy_fft) + 1e-10)
                            clean_mag_db = 20 * np.log10(np.abs(clean_fft) + 1e-10)
                            
                            line_spec_noisy.set_ydata(noisy_mag_db)
                            line_spec_clean.set_ydata(clean_mag_db)
                            
                            # Refresh plot
                            fig.canvas.draw()
                            fig.canvas.flush_events()
                    else:
                        print(f"   ⚠️  Chunk {chunk_num}: No audio decoded")
                else:
                    print(f"   ⚠️  Chunk {chunk_num}: RX failed")
                
                # Minimal delay for faster streaming
                time.sleep(0.05)
            
            # Wait for final audio to finish
            sd.wait()
            
            print(f"\n✅ Real-time streaming complete: {chunk_num} chunks")
            
            # Save full audio if desired
            if audio_buffer:
                full_audio = np.concatenate(audio_buffer)
                output_path = args.output or 'output_realtime_audio.wav'
                from scipy.io import wavfile
                audio_int16 = (full_audio * 32767).astype(np.int16)
                wavfile.write(output_path, modulation.audio_rate, audio_int16)
                print(f"💾 Saved full audio: {output_path}")
            
            # Keep plot open
            print("\n📊 Plot window open. Close to continue...")
            plt.ioff()
            plt.show()
            
        except KeyboardInterrupt:
            print("\n⏹️  Streaming stopped by user")
            sd.stop()
            plt.close()
        
        # Cleanup
        pluto.stop()
        rx_sdr.stop()
        print("\n✅ Real-time session complete")
        return
    
    else:
        # Normal buffered transmission
        pluto.transmit_background(tx_waveform)
    
    print("⏱️  Waiting 2 seconds for TX to stabilize...")
    time.sleep(2)
    
    # ========== STEP 5: Receive ==========
    print("\n📥 STEP 5: Receive Signal")
    print("-" * 80)
    
    rx_waveform = rx_sdr.receive(duration=args.rx_duration)
    
    if rx_waveform is None:
        print("❌ RX failed. Aborting.")
        pluto.stop()
        rx_sdr.stop()
        return
    
    # ========== STEP 6: Demodulate and Denoise ==========
    print("\n🔄 STEP 6: Demodulate and Denoise")
    print("-" * 80)
    
    result = modulation.demodulate(rx_waveform)
    
    # ========== STEP 7: Save Results ==========
    print("\n💾 STEP 7: Save Results")
    print("-" * 80)
    
    # Save denoised audio
    if result['audio'] is not None:
        from scipy.io import wavfile
        output_path = args.output or 'output_audio.wav'
        
        # Convert to int16
        audio_int16 = (result['audio'] * 32767).astype(np.int16)
        wavfile.write(output_path, modulation.audio_rate, audio_int16)
        print(f"💾 Saved audio: {output_path}")
        
        # Play the audio
        print(f"\n🔊 Playing denoised audio...")
        try:
            import sounddevice as sd
            sd.play(result['audio'], modulation.audio_rate)
            print(f"   Playing {len(result['audio'])/modulation.audio_rate:.1f} seconds...")
            sd.wait()  # Wait until playback is finished
            print("✅ Playback complete")
        except Exception as e:
            print(f"⚠️  Playback failed: {e}")
            print(f"   (Audio saved to {output_path})")
    else:
        print("⚠️  No audio to play")
    
    # ========== STEP 8: Cleanup ==========
    print("\n🧹 STEP 8: Cleanup")
    print("-" * 80)
    
    pluto.stop()
    rx_sdr.stop()
    
    print("\n" + "="*80)
    print("✅ FM INFERENCE COMPLETE")
    print("="*80)
    print(f"📂 Plots saved in: src/inference/plot/")
    if args.output:
        print(f"📂 Output saved in: {args.output}")
    print("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
