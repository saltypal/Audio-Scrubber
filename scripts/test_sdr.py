#!/usr/bin/env python
"""Quick test of RTL-SDR with async."""
import sys
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).parent))

async def test():
    try:
        from rtlsdr import RtlSdr
        print("Testing RTL-SDR...")
        sdr = RtlSdr()
        sdr.sample_rate = 240000
        sdr.center_freq = 88.5e6
        print(f"✅ SDR object created")
        print(f"   Sample rate: {sdr.sample_rate}")
        print(f"   Center freq: {sdr.center_freq / 1e6:.2f} MHz")
        
        print("Reading samples...")
        count = 0
        async for samples in sdr.stream(240000):
            print(f"✅ Received {len(samples)} IQ samples")
            count += 1
            if count >= 1:
                break
        
        await sdr.stop()
        print("✅ SUCCESS: RTL-SDR is working!")
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
