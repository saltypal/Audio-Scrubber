import SoapySDR

print("Scanning for devices...")
# 1. List all available devices
results = SoapySDR.Device.enumerate()
print(f"Found {len(results)} device(s).")

for i, dev_args in enumerate(results):
    print(f"\n--- Device {i} ---")
    # Convert the Swig Object to a readable dictionary
    print(dict(dev_args)) 

# 2. Try to open the first one found
if len(results) > 0:
    print("\nAttempting to open Device 0...")
    try:
        # We use the EXACT arguments found in the scan
        sdr = SoapySDR.Device(results[0])
        print("SUCCESS: Device opened!")
        print("Hardware:", sdr.getHardwareInfo())
    except Exception as e:
        print("FAILED to open device:", e)
else:
    print("\nNo devices found (but 'Found Rafael Micro' suggests the driver is working).")