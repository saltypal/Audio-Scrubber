import sys
import time
import numpy as np
import threading
import signal

# GNU Radio Imports
try:
    from gnuradio import gr, digital, blocks, channels, zeromq
except ImportError:
    print("Error: GNU Radio not found. Please ensure you are in an environment with gnuradio installed.")
    sys.exit(1)

class OFDM_Generator(gr.top_block):
    """
    GNU Radio Flowgraph for generating Clean vs Noisy OFDM pairs.
    Streams data via ZeroMQ PUSH sockets.
    """
    def __init__(self):
        gr.top_block.__init__(self, "OFDM Training Generator")

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------
        self.fft_len = 64
        self.cp_len = 16
        self.sample_rate = 1e6
        
        # ---------------------------------------------------------
        # Blocks
        # ---------------------------------------------------------
        
        # 1. Random Source (Infinite stream of random bytes)
        # We create a large buffer of random bytes and repeat it
        random_data = np.random.randint(0, 256, 10000).tolist()
        self.src = blocks.vector_source_b(random_data, True)
        
        # 2. OFDM Transmitter (Hierarchical Block)
        # Handles Modulation, IFFT, Cyclic Prefix, Pilot insertion, etc.
        self.tx = digital.ofdm_tx(
            fft_len=self.fft_len,
            cp_len=self.cp_len,
            packet_len_tags_key="packet_len",
            debug_log=False
        )
        
        # 3. Channel Model (The "Noise Maker")
        # Initial values (will be randomized dynamically)
        self.channel = channels.channel_model(
            noise_voltage=0.1,
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0, 0.1+0.1j], # Simple multipath
            seed=0
        )
        
        # 4. ZeroMQ Sinks (The Output Interface)
        # Port 5555: Clean Data
        # Port 5556: Noisy Data
        self.sink_clean = zeromq.push_sink(gr.sizeof_gr_complex, 1, "tcp://127.0.0.1:5555", 100, False)
        self.sink_noisy = zeromq.push_sink(gr.sizeof_gr_complex, 1, "tcp://127.0.0.1:5556", 100, False)
        
        # ---------------------------------------------------------
        # Connections
        # ---------------------------------------------------------
        # Path 1: Source -> Tx -> Clean Sink
        self.connect(self.src, self.tx, self.sink_clean)
        
        # Path 2: Source -> Tx -> Channel -> Noisy Sink
        self.connect(self.src, self.tx, self.channel, self.sink_noisy)

    def randomize_channel(self):
        """Dynamic parameter updating loop"""
        while True:
            time.sleep(0.2) # Update 5 times a second
            
            # Randomize SNR (Noise Voltage)
            # SNR high -> Low voltage, SNR low -> High voltage
            # Range: 0.01 (Clean) to 0.5 (Very Noisy)
            nv = np.random.uniform(0.01, 0.4)
            
            # Randomize Frequency Offset
            # Range: -0.02 to 0.02 (Normalized frequency)
            fo = np.random.uniform(-0.02, 0.02)
            
            # Apply
            self.channel.set_noise_voltage(nv)
            self.channel.set_frequency_offset(fo)

def main():
    print("Starting GNU Radio OFDM Generator...")
    print("Streaming Clean Data -> tcp://127.0.0.1:5555")
    print("Streaming Noisy Data -> tcp://127.0.0.1:5556")
    
    tb = OFDM_Generator()
    tb.start()
    
    # Start the dynamic randomizer in a background thread
    randomizer = threading.Thread(target=tb.randomize_channel)
    randomizer.daemon = True
    randomizer.start()
    
    def sig_handler(sig, frame):
        print("Stopping flowgraph...")
        tb.stop()
        tb.wait()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        tb.stop()
        tb.wait()

if __name__ == "__main__":
    main()
