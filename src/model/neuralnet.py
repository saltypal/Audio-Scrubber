import torch
from torch import nn

"""
Made by satya sp @ 4:16 15/11/25 with copilot

THIS IS JUST THE STRUCTUREO F THE NURAL NETWOR
I PUT HOW THE LAYERS ARE, HOW THEY CONNECTED AND ALL

1D U-Net for Audio Denoising
- Takes noisy audio input
- Outputs clean audio prediction
- Uses encoder-decoder architecture with skip connections

"""
class Conv1DBlock(nn.Module):
    # this is to do the convolution part 
    # conv1d - > batchnorm -> relu
    #Batch normalization is used to stabilize and speed up the training 
    # of neural networks. 
    # It works by normalizing the output of a layer so that 
    # it has a mean of zero and a standard deviation of one (per batch).

    def __init__(self, in_channels, out_channels, kernel_size = 15, stride = 1, padding=7):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    


class UNet1D(nn.Module):
    """
    1D U-Net for audio denoising.
    
    Architecture:
    - Encoder: Downsamples and extracts features
    - Bottleneck: Deepest layer
    - Decoder: Upsamples and reconstructs clean audio
    - Skip connections: Preserve fine details
    
    Input shape: (batch_size, 1, audio_length)
    Output shape: (batch_size, 1, audio_length)
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(UNet1D, self).__init__()
        
        # --- ENCODER (Downsampling Path) ---
        # Each layer doubles the channels and halves the length
        self.enc1 = Conv1DBlock(in_channels, 16)      # 1 -> 16 channels
        self.pool1 = nn.MaxPool1d(kernel_size=2)       # Downsample by 2
        
        self.enc2 = Conv1DBlock(16, 32)                # 16 -> 32 channels
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.enc3 = Conv1DBlock(32, 64)                # 32 -> 64 channels
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.enc4 = Conv1DBlock(64, 128)               # 64 -> 128 channels
        self.pool4 = nn.MaxPool1d(kernel_size=2)
        

         # --- BOTTLENECK ---
        self.bottleneck = Conv1DBlock(128, 256)        # 128 -> 256 channels
        
        # --- DECODER (Upsampling Path) ---
        # Each layer halves the channels and doubles the length
        self.upconv4 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec4 = Conv1DBlock(256, 128)  # 256 because of skip connection (128 + 128)
        
        self.upconv3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec3 = Conv1DBlock(128, 64)   # 128 because of skip connection (64 + 64)
        
        self.upconv2 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec2 = Conv1DBlock(64, 32)    # 64 because of skip connection (32 + 32)
        
        self.upconv1 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.dec1 = Conv1DBlock(32, 16)    # 32 because of skip connection (16 + 16)
        
        # --- OUTPUT LAYER ---
        # Final 1x1 convolution to get back to 1 channel (clean audio)
        self.out = nn.Conv1d(16, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, 1, audio_length)
        
        Returns:
            Output tensor of shape (batch_size, 1, audio_length)
        """
        # --- ENCODER ---
        enc1 = self.enc1(x)           # Save for skip connection
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)           # Save for skip connection
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)           # Save for skip connection
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)           # Save for skip connection
        x = self.pool4(enc4)
        
        # --- BOTTLENECK ---
        x = self.bottleneck(x)
        
        # --- DECODER ---
        # Upsample and concatenate with encoder features (skip connections)
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)  # Concatenate along channel dimension
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # --- OUTPUT ---
        x = self.out(x)
        
        return x
    

# --- Quick Test ---
if __name__ == "__main__":
    # Create a dummy input (batch_size=2, channels=1, length=16000)
    # Simulates 2 audio clips of ~0.7 seconds at 22050 Hz
    dummy_input = torch.randn(2, 1, 16000)
    
    # Create the model
    model = UNet1D(in_channels=1, out_channels=1)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\n✅ 1D U-Net is working correctly!")