import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ofdm.model.neuralnet import OFDM_UNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import zmq
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ofdm.model.neuralnet import OFDM_UNet

class ZMQDataset(IterableDataset):
    def __init__(self, clean_addr="tcp://127.0.0.1:5555", noisy_addr="tcp://127.0.0.1:5556", chunk_size=1024):
        self.clean_addr = clean_addr
        self.noisy_addr = noisy_addr
        self.chunk_size = chunk_size
        self.context = zmq.Context()

    def __iter__(self):
        clean_socket = self.context.socket(zmq.SUB)
        clean_socket.connect(self.clean_addr)
        clean_socket.setsockopt(zmq.SUBSCRIBE, b"")

        noisy_socket = self.context.socket(zmq.SUB)
        noisy_socket.connect(self.noisy_addr)
        noisy_socket.setsockopt(zmq.SUBSCRIBE, b"")

        print("Waiting for GNU Radio stream...")
        
        while True:
            # Receive raw bytes
            clean_bytes = clean_socket.recv()
            noisy_bytes = noisy_socket.recv()

            # Convert to complex64
            clean_data = np.frombuffer(clean_bytes, dtype=np.complex64)
            noisy_data = np.frombuffer(noisy_bytes, dtype=np.complex64)

            # Ensure we have enough data
            min_len = min(len(clean_data), len(noisy_data))
            num_chunks = min_len // self.chunk_size

            for i in range(num_chunks):
                start = i * self.chunk_size
                end = start + self.chunk_size
                
                c_chunk = clean_data[start:end]
                n_chunk = noisy_data[start:end]

                # Convert to Tensor [2, 1024] (Real, Imag)
                clean_tensor = torch.stack([torch.from_numpy(c_chunk.real), torch.from_numpy(c_chunk.imag)])
                noisy_tensor = torch.stack([torch.from_numpy(n_chunk.real), torch.from_numpy(n_chunk.imag)])

                yield noisy_tensor.float(), clean_tensor.float()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Model
    model = OFDM_UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Initialize Dataset
    dataset = ZMQDataset()
    dataloader = DataLoader(dataset, batch_size=32)

    print("Starting Training... (Ensure GNU Radio Flowgraph is running!)")
    
    model.train()
    step = 0
    
    try:
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}")
                
            if step % 1000 == 0:
                save_path = Path("saved_models/OFDM/ofdm_unet_latest.pth")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")

    except KeyboardInterrupt:
        print("Training stopped.")

if __name__ == "__main__":
    train()

if __name__ == "__main__":
    train()
