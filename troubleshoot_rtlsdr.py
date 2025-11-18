"""
RTL-SDR Troubleshooting Script
Diagnoses RTL-SDR hardware and driver issues
"""
import sys
from pathlib import Path

print("\n" + "="*70)
print("RTL-SDR TROUBLESHOOTING DIAGNOSTIC")
print("="*70 + "\n")

# Step 0: Check Python environment
print("[0] Checking Python environment...")
import os
print(f"    Python executable: {sys.executable}")
print(f"    Python version: {sys.version.split()[0]}")
print(f"    Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')}")

# Step 1: Check pyrtlsdr installation
print("\n[1] Checking pyrtlsdr installation...")
try:
    from rtlsdr import RtlSdr
    import rtlsdr
    print("    [OK] pyrtlsdr is installed")
    print(f"    [OK] Version: {rtlsdr.__version__ if hasattr(rtlsdr, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"    [FAIL] FAILED: {e}")
    print("    Install with: pip install pyrtlsdr")
    sys.exit(1)

# Step 2: Check libusb
print("\n[2] Checking libusb...")
try:
    import usb1
    print("    [OK] libusb is available")
except ImportError:
    print("    [FAIL] libusb not found (pyUSB)")
    print("    Install with: pip install pyusb")

# Step 3: Try to detect RTL-SDR devices
print("\n[3] Scanning for RTL-SDR devices...")
try:
    import usb.core
    devices = usb.core.find(find_all=True, idVendor=0x0bda, idProduct=0x2838)
    device_list = list(devices)
    
    if device_list:
        print(f"    [OK] Found {len(device_list)} RTL-SDR device(s)")
        for i, dev in enumerate(device_list):
            print(f"      Device {i}: {dev.manufacturer} {dev.product}")
    else:
        print("    [FAIL] No RTL-SDR devices found via USB")
        print("    -> Check if device is connected")
        print("    -> Try different USB port")
except Exception as e:
    print(f"    [FAIL] USB scan failed: {e}")

# Step 4: Check device permissions
print("\n[4] Testing RTL-SDR device access...")
try:
    from rtlsdr.rtlsdraio import RtlSdrAio
    import asyncio
    
    async def test_sdr():
        sdr = RtlSdrAio()
        await sdr.open()
        print("    [OK] RTL-SDR device accessible!")
        print(f"    [OK] Sample rate: {sdr.sample_rate}")
        print(f"    [OK] Gain values: {sdr.gain_values}")
        await sdr.stop()
        sdr.close()
    
    # Run async test
    asyncio.run(test_sdr())
    
except Exception as e:
    error_str = str(e)
    print(f"    [FAIL] FAILED: {error_str}")
    
    if "Access denied" in error_str or "LIBUSB_ERROR_ACCESS" in error_str:
        print("\n    [INFO] ACCESS DENIED - This is a driver issue!")
        print("    Solution: Reinstall Zadig driver with WinUSB")
        print("    Steps:")
        print("      1. Download Zadig from: https://zadig.akeo.ie/")
        print("      2. Connect RTL-SDR device")
        print("      3. Run Zadig as Administrator")
        print("      4. Options -> List All Devices")
        print("      5. Find 'Realtek RTL2832U EEPROM' or 'RTL2832 EEPROM'")
        print("      6. Driver: WinUSB -> Install Driver")
        print("      7. Restart this script")
    elif "No such device" in error_str or "device index" in error_str:
        print("\n    [INFO] DEVICE NOT FOUND - Check connection!")
        print("    Steps:")
        print("      1. Unplug RTL-SDR from all USB ports")
        print("      2. Wait 5 seconds")
        print("      3. Plug into a different USB port (preferably USB 2.0)")
        print("      4. Wait for driver to load")
        print("      5. Run this script again")
    elif "Could not open SDR" in error_str:
        print("\n    [INFO] DEVICE BUSY OR PERMISSION ISSUE")
        print("    Possible causes:")
        print("      - Another program is using the RTL-SDR")
        print("      - Windows needs administrator privileges")
        print("      - Conda environment doesn't have proper libusb")
        print("    Solutions:")
        print("      1. Close any SDR software (SDR#, HDSDR, etc.)")
        print("      2. Try running as Administrator")
        print("      3. Reinstall pyrtlsdr: conda activate AScrubber && pip uninstall pyrtlsdr && pip install pyrtlsdr")
        print("      4. Try in base environment: conda activate base && python troubleshoot_rtlsdr.py")
    else:
        print(f"\n    [INFO] Unknown error: {error_str}")
        print("    Try:")
        print("      - Restart computer")
        print("      - Reinstall pyrtlsdr: pip uninstall pyrtlsdr && pip install pyrtlsdr")
        print("      - Check if device works in base conda environment")

# Step 5: Check configuration files
print("\n[5] Checking configuration...")
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import AudioSettings, RTLSDRSettings, Paths
    print("    [OK] config.py loaded successfully")
    print(f"    [OK] Sample rate: {AudioSettings.SAMPLE_RATE}")
    print(f"    [OK] Model path: {Paths.MODEL_BEST}")
except Exception as e:
    print(f"    [FAIL] Config error: {e}")

# Step 6: Check saved model
print("\n[6] Checking AI model...")
try:
    model_path = Path(__file__).parent / "saved_models" / "FM" / "unet1d_best.pth"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024*1024)
        print(f"    [OK] Model found: {model_path}")
        print(f"    [OK] Size: {size_mb:.1f} MB")
    else:
        print(f"    [WARN] Model not found at: {model_path}")
except Exception as e:
    print(f"    [FAIL] Model check failed: {e}")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70 + "\n")
