"""
================================================================================
PLUTO → RTL-SDR PIPELINE TEST
================================================================================
Quick test script for the clean OFDM transmission pipeline.

Usage:
    # Full loopback test
    python TEST_PIPELINE.py
    
    # TX only
    python TEST_PIPELINE.py --tx-only --message "Custom message"
    
    # RX only
    python TEST_PIPELINE.py --rx-only --duration 5
    
    # Change frequency
    python TEST_PIPELINE.py --freq 433e6
================================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sdr_hardware_clean import PlutoTX, RTLSDR_RX, run_loopback, SDR_CONFIG


def test_tx_only(message="Test from Pluto"):
    """Test transmitter only."""
    print("="*80)
    print("TRANSMITTER TEST")
    print("="*80)
    
    tx = PlutoTX()
    
    if not tx.connect():
        print("❌ Failed to connect to Pluto")
        return False
    
    if not tx.configure():
        print("❌ Failed to configure Pluto")
        return False
    
    tx.transmit_message(message, plot=True)
    return True


def test_rx_only(duration=5):
    """Test receiver only."""
    print("="*80)
    print("RECEIVER TEST")
    print("="*80)
    
    rx = RTLSDR_RX()
    
    if not rx.connect():
        print("❌ Failed to connect to RTL-SDR")
        return False
    
    if not rx.configure():
        print("❌ Failed to configure RTL-SDR")
        return False
    
    result = rx.receive_and_decode(duration=duration, plot=True)
    rx.close()
    
    return result is not None


def test_loopback():
    """Test full Pluto → RTL-SDR loopback."""
    messages = [
        "Hello OFDM!",
        "Testing 123",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "The quick brown fox jumps over the lazy dog",
    ]
    
    print("="*80)
    print("LOOPBACK TEST - MULTIPLE MESSAGES")
    print("="*80)
    
    results = []
    for i, msg in enumerate(messages, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(messages)}")
        print(f"{'='*80}")
        
        success = run_loopback(msg, duration=3, add_noise=False)
        results.append((msg, success))
        
        if i < len(messages):
            print("\n⏳ Waiting 3s before next test...")
            import time
            time.sleep(3)
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    for msg, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - '{msg}'")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test OFDM SDR Pipeline')
    parser.add_argument('--tx-only', action='store_true', help='Test TX only')
    parser.add_argument('--rx-only', action='store_true', help='Test RX only')
    parser.add_argument('--message', type=str, default='Test from Pluto', help='Message for TX')
    parser.add_argument('--duration', type=int, default=5, help='RX duration (seconds)')
    parser.add_argument('--freq', type=float, help='Center frequency (Hz)')
    parser.add_argument('--tx-gain', type=int, help='TX gain (dB)')
    parser.add_argument('--rx-gain', help='RX gain (auto or dB value)')
    
    args = parser.parse_args()
    
    # Update config
    if args.freq:
        SDR_CONFIG['CENTER_FREQ'] = args.freq
        print(f"📡 Center Freq: {args.freq/1e6:.1f} MHz")
    
    if args.tx_gain is not None:
        SDR_CONFIG['TX_GAIN'] = args.tx_gain
        print(f"📡 TX Gain: {args.tx_gain} dB")
    
    if args.rx_gain:
        SDR_CONFIG['RX_GAIN'] = args.rx_gain
        print(f"📡 RX Gain: {args.rx_gain}")
    
    # Run tests
    if args.tx_only:
        test_tx_only(args.message)
    elif args.rx_only:
        test_rx_only(args.duration)
    else:
        test_loopback()
