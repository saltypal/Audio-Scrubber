
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ofdm.model.ofdm_engine import OFDMModulator, OFDMDemodulator, QPSKMapper, OFDMConfig

def debug_engine():
    print("Debugging OFDM Engine...")
    
    # 1. Test QPSK Mapping
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    mapper = QPSKMapper()
    syms = mapper.bits_to_qpsk(bits)
    print(f"Bits: {bits}")
    print(f"Syms: {syms}")
    
    rec_bits = mapper.qpsk_to_bits(syms)
    print(f"Rec Bits: {rec_bits}")
    assert np.array_equal(bits, rec_bits), "QPSK Mapping failed"
    print("QPSK Mapping OK")
    
    # 2. Test Modulator -> Demodulator (No Scaling)
    mod = OFDMModulator()
    demod = OFDMDemodulator()
    
    # Create random symbols
    # 8 data carriers per symbol. Let's do 1 symbol.
    data_syms = np.array([1+1j, -1+1j, -1-1j, 1-1j, 1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64)
    
    waveform = mod.modulate_symbols(data_syms)
    print(f"Waveform power: {np.mean(np.abs(waveform)**2)}")
    
    rec_syms = demod.demodulate_waveform(waveform)
    
    print("\nOriginal vs Received Symbols:")
    for i in range(len(data_syms)):
        print(f"Orig: {data_syms[i]:.2f}, Recv: {rec_syms[i]:.2f}, Ratio: {rec_syms[i]/data_syms[i]:.2f}")
        
    # Check phase
    orig_phase = np.angle(data_syms)
    recv_phase = np.angle(rec_syms)
    print(f"\nPhase Diff (deg): {np.degrees(recv_phase - orig_phase)}")
    
    # Check bits
    rec_bits_from_wave = mapper.qpsk_to_bits(rec_syms)
    orig_bits_from_syms = mapper.qpsk_to_bits(data_syms)
    
    print(f"\nOrig Bits: {orig_bits_from_syms}")
    print(f"Recv Bits: {rec_bits_from_wave}")
    
    if np.array_equal(orig_bits_from_syms, rec_bits_from_wave):
        print("✅ Loopback Test PASSED")
    else:
        print("❌ Loopback Test FAILED")

if __name__ == "__main__":
    debug_engine()
