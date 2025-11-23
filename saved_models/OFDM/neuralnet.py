import torch
import torch.nn as nn

class Conv1DBlock(nn.Module):
    """
    Basic Convolutional Block: Conv1d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class OFDM_UNet(nn.Module):
    """
    1D U-Net for OFDM IQ Denoising.
    
    Input:  (Batch, 2, Length) -> [I, Q] channels
    Output: (Batch, 2, Length) -> [I, Q] cleaned channels
    """
    def __init__(self, in_channels=2, out_channels=2):
        super(OFDM_UNet, self).__init__()
        
        # --- ENCODER ---
        self.enc1 = Conv1DBlock(in_channels, 32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = Conv1DBlock(32, 64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.enc3 = Conv1DBlock(64, 128)
        self.pool3 = nn.MaxPool1d(2)
        
        self.enc4 = Conv1DBlock(128, 256)
        self.pool4 = nn.MaxPool1d(2)
        
        # --- BOTTLENECK ---
        self.bottleneck = Conv1DBlock(256, 512)
        
        # --- DECODER ---
        self.upconv4 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec4 = Conv1DBlock(512, 256) # 256+256 inputs
        
        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec3 = Conv1DBlock(256, 128)
        
        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2 = Conv1DBlock(128, 64)
        
        self.upconv1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec1 = Conv1DBlock(64, 32)
        
        # --- OUTPUT ---
        self.out = nn.Conv1d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)

if __name__ == "__main__":
    # Test
    model = OFDM_UNet()
    x = torch.randn(1, 2, 1024) # Batch 1, I/Q, 1024 samples
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
