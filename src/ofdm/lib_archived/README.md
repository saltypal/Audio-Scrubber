
# Robust OFDM Library

This library provides a modular, correct implementation of an OFDM Transmitter and Receiver, compatible with the GNU Radio dataset format used for training.

## Structure

- **`config.py`**: System parameters (FFT size, CP length, Pilot/Data carriers). Matches `dataset_gnu.py`.
- **`modulation.py`**: QPSK modulation and demodulation (Correctly mapped).
- **`core.py`**: Low-level OFDM operations (IFFT/FFT, Cyclic Prefix, Resource Mapping).
- **`receiver.py`**: Receiver logic including **Channel Equalization** using pilots.
- **`transceiver.py`**: High-level `OFDMTransmitter` and `OFDMReceiver` classes.
- **`utils.py`**: Visualization and helper functions.

## Usage

```python
from src.ofdm.lib.transceiver import OFDMTransmitter, OFDMReceiver

# 1. Transmit
tx = OFDMTransmitter()
text = "Hello World"
waveform, meta = tx.transmit(text.encode('utf-8'))

# 2. Receive
rx = OFDMReceiver()
payload, info = rx.receive(waveform)
print(payload.decode('utf-8'))
```

## Features

- **Robust Synchronization**: Uses length header (simple) but relies on correct frame alignment (assumes start of buffer is start of frame).
- **Channel Equalization**: Uses Pilot subcarriers to estimate and correct phase/amplitude distortions (essential for AI-denoised signals).
- **Correct Scaling**: Normalizes power to ~35.0 to match the AI model's training distribution.
