import torch
from torch import nn

"""
Created by Satya with Copilot @ 16/11/25

STFT-based Conv2D U-Net for Audio Denoising
- Takes STFT magnitude spectrogram as input
- Uses 2D convolutions to process time-frequency domain
- Outputs clean STFT magnitude prediction
- Better for capturing spectral patterns than 1D time-domain models

Architecture:
- Input: (batch, 1, freq_bins, time_frames) - STFT magnitude spectrogram
- Encoder: 4 layers, 2D convolutions with skip connections
- Decoder: 4 layers, 2D transpose convolutions
- Output: (batch, 1, freq_bins, time_frames) - denoised spectrogram

Key Differences from 1D U-Net:
- Works in frequency domain (STFT) instead of time domain
- Better at removing stationary noise (hum, hiss)
- Preserves phase information by using magnitude + phase reconstruction
"""

class Conv2DBlock(nn.Module):
    """
    2D Convolution Block: Conv2D -> BatchNorm2D -> ReLU
    
    Used in both encoder and decoder paths.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UNet2D(nn.Module):
    """
    2D U-Net for STFT-based audio denoising.
    
    Architecture:
    - Encoder: Downsamples and extracts features from spectrogram
    - Bottleneck: Deepest layer for feature compression
    - Decoder: Upsamples and reconstructs clean spectrogram
    - Skip connections: Preserve fine spectral details
    
    Input shape: (batch_size, 1, freq_bins, time_frames)
    - freq_bins: typically 1024 for STFT (n_fft/2 + 1)
    - time_frames: variable depending on audio length
    
    Output shape: (batch_size, 1, freq_bins, time_frames)
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(UNet2D, self).__init__()
        
        # --- ENCODER (Downsampling Path) ---
        # Kernel size 3, padding 1 to preserve dimensions before pooling
        self.enc1 = Conv2DBlock(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # Downsample by 2 in both dimensions
        
        self.enc2 = Conv2DBlock(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.enc3 = Conv2DBlock(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.enc4 = Conv2DBlock(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # --- BOTTLENECK ---
        self.bottleneck = Conv2DBlock(128, 256, kernel_size=3, padding=1)
        
        # --- DECODER (Upsampling Path) ---
        # Each layer halves the channels and doubles the spatial dimensions
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = Conv2DBlock(256, 128, kernel_size=3, padding=1)  # 256 = 128 (upsampled) + 128 (skip)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = Conv2DBlock(128, 64, kernel_size=3, padding=1)   # 128 = 64 (upsampled) + 64 (skip)
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = Conv2DBlock(64, 32, kernel_size=3, padding=1)    # 64 = 32 (upsampled) + 32 (skip)
        
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = Conv2DBlock(32, 16, kernel_size=3, padding=1)    # 32 = 16 (upsampled) + 16 (skip)
        
        # --- OUTPUT LAYER ---
        # Final 1x1 convolution to get back to 1 channel (clean spectrogram)
        self.out = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the 2D U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, 1, freq_bins, time_frames)
        
        Returns:
            Output tensor of shape (batch_size, 1, freq_bins, time_frames)
        """
        # --- ENCODER ---
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # --- BOTTLENECK ---
        x = self.bottleneck(x)
        
        # --- DECODER ---
        # Upsample and concatenate with encoder features (skip connections)
        # Pad if necessary to match sizes
        x = self.upconv4(x)
        if x.shape != enc4.shape:
            # Pad to match encoder shape
            pad_h = enc4.shape[2] - x.shape[2]
            pad_w = enc4.shape[3] - x.shape[3]
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        if x.shape != enc3.shape:
            pad_h = enc3.shape[2] - x.shape[2]
            pad_w = enc3.shape[3] - x.shape[3]
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        if x.shape != enc2.shape:
            pad_h = enc2.shape[2] - x.shape[2]
            pad_w = enc2.shape[3] - x.shape[3]
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        if x.shape != enc1.shape:
            pad_h = enc1.shape[2] - x.shape[2]
            pad_w = enc1.shape[3] - x.shape[3]
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # --- OUTPUT ---
        x = self.out(x)
        
        # Ensure output matches input size
        if x.shape != enc1.shape:
            # Crop or pad to match input shape
            pad_h = enc1.shape[2] - x.shape[2]
            pad_w = enc1.shape[3] - x.shape[3]
            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, max(0, pad_w), 0, max(0, pad_h)))
            if pad_h < 0 or pad_w < 0:
                x = x[:, :, :enc1.shape[2], :enc1.shape[3]]
        
        return x


# --- Quick Test ---
if __name__ == "__main__":
    # Create a dummy STFT spectrogram input
    # Shape: (batch_size=2, channels=1, freq_bins=1024, time_frames=256)
    # This represents ~5.8 seconds at 22050 Hz with hop_length=256
    dummy_input = torch.randn(2, 1, 1024, 256)
    
    # Create the model
    model = UNet2D(in_channels=1, out_channels=1)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\n✅ 2D U-Net for STFT is working correctly!")
