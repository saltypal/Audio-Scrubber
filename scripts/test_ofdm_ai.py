"""
Standalone OFDM AI denoising tester (NO HARDWARE REQUIRED).
Workflow:
  1. Load data file (image/binary)
  2. Modulate via existing OFDM_Modulation (transmitter path)
  3. Apply synthetic channel impairments (AWGN, CFO, multipath, IQ imbalance, phase noise)
  4. Run AI denoising (if model provided / auto-detected)
  5. Demodulate both noisy and AI-denoised waveforms using OFDM receiver path
  6. Compute BER, SNR improvement, image reconstruction & comparison plots

Usage examples:
  python scripts/test_ofdm_ai.py --data src/inference/TxRx/content/testfile.png --model saved_models/OFDM/final_models/ofdm_1dunet_best_fixed.pth
  python scripts/test_ofdm_ai.py --data my.bin --snr-db 8 --cfo 150 --multipath-taps 4 --phase-noise 0.003

If --model is omitted it will attempt auto-discovery like the main inference script.
"""
import argparse
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Ensure src is in path
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference.TxRx.ofdm_modulation import OFDM_Modulation
from inference.TxRx.sdr_utils import SDRUtils


def apply_channel_impairments(waveform: np.ndarray,
                              snr_db: float = 12.0,
                              cfo_hz: float = 0.0,
                              sample_rate: float = 2e6,
                              multipath_taps: int = 0,
                              iq_imbalance: float = 0.0,
                              phase_noise_std: float = 0.0) -> np.ndarray:
    """Apply synthetic impairments to OFDM waveform.
    Args:
        waveform: Clean complex IQ samples
        snr_db: Target SNR after AWGN
        cfo_hz: Carrier frequency offset in Hz
        sample_rate: Sample rate for CFO phase progression
        multipath_taps: Number of extra delayed taps (random small amplitudes)
        iq_imbalance: Relative amplitude imbalance between I and Q (e.g., 0.1 = 10%)
        phase_noise_std: Std dev of cumulative phase noise Gaussian increments
    Returns:
        Impaired waveform (complex64)
    """
    impaired = waveform.astype(np.complex64).copy()

    # IQ imbalance
    if iq_imbalance != 0.0:
        i = np.real(impaired)
        q = np.imag(impaired)
        i *= (1 + iq_imbalance / 2)
        q *= (1 - iq_imbalance / 2)
        impaired = i + 1j * q

    # Multipath (FIR random taps)
    if multipath_taps > 0:
        taps = [1.0]
        for _ in range(multipath_taps):
            taps.append(0.3 * (np.random.randn() + 1j * np.random.randn()))
        taps = np.array(taps, dtype=np.complex64)
        impaired = np.convolve(impaired, taps, mode='same')

    # CFO (Carrier frequency offset)
    if cfo_hz != 0.0:
        n = np.arange(len(impaired))
        phase_rot = np.exp(1j * 2 * np.pi * cfo_hz * n / sample_rate)
        impaired *= phase_rot

    # Phase noise
    if phase_noise_std > 0.0:
        increments = np.random.randn(len(impaired)) * phase_noise_std
        phase = np.cumsum(increments)
        impaired *= np.exp(1j * phase)

    # AWGN
    if snr_db is not None:
        sig_power = np.mean(np.abs(impaired) ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = (np.random.randn(len(impaired)) + 1j * np.random.randn(len(impaired))) * np.sqrt(noise_power / 2)
        impaired += noise

    return impaired.astype(np.complex64)


def compute_snr(clean: np.ndarray, test: np.ndarray) -> float:
    """Compute SNR between clean reference and test signal."""
    noise = test - clean
    sig_p = np.mean(np.abs(clean) ** 2)
    noise_p = np.mean(np.abs(noise) ** 2) + 1e-12
    return 10 * np.log10(sig_p / noise_p)


def plot_results(original_img, noisy_img, ai_img, clean_waveform, noisy_waveform, ai_waveform,
                 output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # Row 1: Images
    ax1 = fig.add_subplot(gs[0, 0])
    if original_img is not None:
        ax1.imshow(original_img)
    ax1.set_title('Original Image', fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    if noisy_img is not None:
        ax2.imshow(noisy_img)
    ax2.set_title('Noisy Reconstructed', fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    if ai_img is not None:
        ax3.imshow(ai_img)
    ax3.set_title('AI Denoised Reconstructed', fontweight='bold')
    ax3.axis('off')

    # Constellations
    def extract_symbols(waveform, cfg):
        fft = cfg.fft_size
        cp = cfg.cp_len
        symbol_len = fft + cp
        num_symbols = len(waveform) // symbol_len
        all_syms = []
        data_idx = cfg.data_carriers
        for i in range(num_symbols):
            sym = waveform[i*symbol_len:(i+1)*symbol_len]
            sym_no_cp = sym[cp:]
            fd = np.fft.fft(sym_no_cp)
            all_syms.append(fd[data_idx])
        return np.concatenate(all_syms) if all_syms else np.array([])

    cfg = OFDM_Modulation(use_ai=False).config  # temp instance to get config
    noisy_syms = extract_symbols(noisy_waveform, cfg)
    ai_syms = extract_symbols(ai_waveform, cfg)

    qpsk_ref = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)

    ax4 = fig.add_subplot(gs[1, 0])
    if noisy_syms.size > 0:
        ax4.scatter(noisy_syms.real, noisy_syms.imag, s=5, alpha=0.3, c='orange', label='Noisy')
    ax4.scatter(qpsk_ref.real, qpsk_ref.imag, c='red', marker='x', s=120, linewidths=3, label='QPSK')
    ax4.set_title('Constellation BEFORE AI', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    ax4.set_xlim(-1.6, 1.6)
    ax4.set_ylim(-1.6, 1.6)
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    if ai_syms.size > 0:
        ax5.scatter(ai_syms.real, ai_syms.imag, s=5, alpha=0.3, c='blue', label='AI Denoised')
    ax5.scatter(qpsk_ref.real, qpsk_ref.imag, c='red', marker='x', s=120, linewidths=3, label='QPSK')
    ax5.set_title('Constellation AFTER AI', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    ax5.set_xlim(-1.6, 1.6)
    ax5.set_ylim(-1.6, 1.6)
    ax5.legend()

    # Waveform & PSD
    ax6 = fig.add_subplot(gs[1, 2])
    t = np.arange(2000)
    ax6.plot(np.real(noisy_waveform[:2000]), label='Noisy I', alpha=0.5, linewidth=0.8)
    ax6.plot(np.real(ai_waveform[:2000]), label='AI I', alpha=0.8, linewidth=0.8)
    ax6.set_title('I Channel (First 2000 samples)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 0])
    ax7.psd(noisy_waveform, NFFT=1024, Fs=2.0, scale_by_freq=False, color='orange', alpha=0.7)
    ax7.psd(ai_waveform, NFFT=1024, Fs=2.0, scale_by_freq=False, color='blue', alpha=0.7)
    ax7.set_title('PSD Comparison')
    ax7.set_xlabel('Frequency (MHz)')
    ax7.set_ylabel('Power')
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])
    # Error magnitude vs clean waveform (for first 2000 samples)
    err_noisy = np.abs(noisy_waveform[:2000] - clean_waveform[:2000])
    err_ai = np.abs(ai_waveform[:2000] - clean_waveform[:2000])
    ax8.plot(err_noisy, label='Noisy Error', color='orange', alpha=0.6)
    ax8.plot(err_ai, label='AI Error', color='blue', alpha=0.6)
    ax8.set_title('Error Magnitude (First 2000 samples)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.text(0.02, 0.95, 'SNR Metrics:', fontweight='bold')
    snr_before = compute_snr(clean_waveform, noisy_waveform)
    snr_after = compute_snr(clean_waveform, ai_waveform)
    ax9.text(0.02, 0.80, f'Before AI SNR: {snr_before:.2f} dB')
    ax9.text(0.02, 0.70, f'After AI SNR: {snr_after:.2f} dB')
    ax9.text(0.02, 0.60, f'Improvement: {snr_after - snr_before:.2f} dB')
    ax9.axis('off')

    plt.suptitle('OFDM AI Denoising (Synthetic Channel Test)', fontsize=16, fontweight='bold')
    out_path = output_dir / 'ofdm_ai_test_results.png'
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f'📊 Saved results plot: {out_path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='OFDM AI Denoising Test (No Hardware)')
    parser.add_argument('--data', required=True, help='Path to data file (image/bin)')
    parser.add_argument('--model', help='Path to AI model (.pth)')
    parser.add_argument('--snr-db', type=float, default=12.0, help='AWGN SNR (dB)')
    parser.add_argument('--cfo', type=float, default=0.0, help='Carrier frequency offset (Hz)')
    parser.add_argument('--multipath-taps', type=int, default=2, help='Additional random multipath taps')
    parser.add_argument('--iq-imbalance', type=float, default=0.0, help='Relative IQ amplitude imbalance (e.g. 0.1 = 10%)')
    parser.add_argument('--phase-noise', type=float, default=0.0, help='Std dev of cumulative phase noise increments')
    parser.add_argument('--output-dir', default='output/ofdm_ai_test', help='Directory to store results')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI denoising (control only)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # 1. Load data
    data_bytes = SDRUtils.load_data(args.data)

    # 2. Modulate (AI off just for clean baseline generation)
    modulation_clean = OFDM_Modulation(use_ai=False)
    tx_waveform = modulation_clean.modulate(data_bytes, image_path=args.data)

    # 3. Apply synthetic channel impairments
    noisy_waveform = apply_channel_impairments(
        tx_waveform,
        snr_db=args.snr_db,
        cfo_hz=args.cfo,
        sample_rate=modulation_clean.config.sample_rate,
        multipath_taps=args.multipath_taps,
        iq_imbalance=args.iq_imbalance,
        phase_noise_std=args.phase_noise
    )
    print(f'✅ Applied synthetic channel impairments')

    # 4. Prepare AI modulation instance (with model)
    modulation_ai = OFDM_Modulation(use_ai=not args.no_ai, model_path=args.model)

    # 5. Demodulate with control & AI
    print('\n🔄 Demodulating NOISY waveform (Control + AI)...')
    result = modulation_ai.demodulate(noisy_waveform)

    control_bytes = result.get('control_data')
    ai_bytes = result.get('data') if modulation_ai.use_ai else None

    # 6. Reconstruct images if original was image
    original_img = None
    noisy_img = None
    ai_img = None
    suffix = Path(args.data).suffix.lower()
    if suffix in ['.png', '.jpg', '.jpeg', '.bmp'] and control_bytes is not None:
        from PIL import Image
        import io
        try:
            original_img = Image.open(args.data)
            noisy_img = Image.open(io.BytesIO(control_bytes)) if control_bytes else None
            ai_img = Image.open(io.BytesIO(ai_bytes)) if ai_bytes else None
        except Exception as e:
            print(f'⚠️ Image reconstruction failed: {e}')

    # 7. Plot results (handle possible None denoised waveform explicitly to avoid ambiguous truth value)
    denoised_waveform = result.get('waveform_denoised')
    if denoised_waveform is None:
        denoised_waveform = noisy_waveform
    plot_results(original_img, noisy_img, ai_img,
                 tx_waveform, noisy_waveform, denoised_waveform,
                 output_dir)

    # 8. Metrics
    if control_bytes and ai_bytes and control_bytes != ai_bytes:
        match_len = min(len(control_bytes), len(ai_bytes))
        errors = np.sum(np.frombuffer(control_bytes[:match_len], dtype=np.uint8) !=
                        np.frombuffer(ai_bytes[:match_len], dtype=np.uint8))
        ber = errors / (match_len * 8)
        print(f'📊 AI vs Control BER (byte-level proxy): {ber:.6f} ({errors} differing bytes of {match_len})')
    else:
        print('⚠️ BER comparison unavailable (missing or identical data)')

    print('\n✅ Test complete (no hardware required). Output dir:', output_dir)


if __name__ == '__main__':
    main()
