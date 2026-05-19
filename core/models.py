import torch
import torch.nn as nn

# --- Helper Block: Double Convolution ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

# --- Architecture 1 & 2: Standard U-Net (Baseline) ---
class UnetBaseline(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.conv_up1(torch.cat([self.up1(x4), x3], dim=1))
        x = self.conv_up2(torch.cat([self.up2(x), x2], dim=1))
        x = self.conv_up3(torch.cat([self.up3(x), x1], dim=1))
        return self.out(x)

# --- Architecture 3 & 5: Decoupled U-Net ---
class UnetDecoupled(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2); self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2); self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2); self.conv_up3 = DoubleConv(128, 64)
        
        self.out_h = nn.Conv2d(64, 1, kernel_size=1)
        self.out_v = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.conv_up1(torch.cat([self.up1(x4), x3], dim=1))
        x = self.conv_up2(torch.cat([self.up2(x), x2], dim=1))
        x = self.conv_up3(torch.cat([self.up3(x), x1], dim=1))
        return torch.cat([self.out_h(x), self.out_v(x)], dim=1)

# --- Architecture 4: Attention U-Net (CBAM) ---
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(nn.functional.adaptive_avg_pool2d(x, 1))
        max_out = self.fc(nn.functional.adaptive_max_pool2d(x, 1))
        x = x * self.sigmoid(avg_out + max_out)
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return x

class UnetAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64); self.cbam1 = CBAM(64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128)); self.cbam2 = CBAM(128)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)); self.cbam3 = CBAM(256)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2); self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2); self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2); self.conv_up3 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        x1 = self.cbam1(self.inc(x))
        x2 = self.cbam2(self.down1(x1))
        x3 = self.cbam3(self.down2(x2))
        x4 = self.down3(x3)
        x = self.conv_up1(torch.cat([self.up1(x4), x3], dim=1))
        x = self.conv_up2(torch.cat([self.up2(x), x2], dim=1))
        x = self.conv_up3(torch.cat([self.up3(x), x1], dim=1))
        return self.out(x)

# --- Architecture 6: Inception U-Net ---
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c = out_channels // 4
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(in_channels, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                                nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_channels, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                                nn.Conv2d(c, c, 5, padding=2), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_channels, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        
    def forward(self, x): return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class InceptionUNet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.inc = nn.Sequential(nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.down1 = nn.Sequential(nn.MaxPool2d(2), InceptionModule(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), InceptionModule(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), InceptionModule(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.conv_up1 = InceptionModule(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.conv_up2 = InceptionModule(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2); self.conv_up3 = InceptionModule(128, 64)
        self.out_h = nn.Conv2d(64, 1, 1); self.out_v = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.conv_up1(torch.cat([self.up1(x4), x3], dim=1))
        x = self.conv_up2(torch.cat([self.up2(x), x2], dim=1))
        x = self.conv_up3(torch.cat([self.up3(x), x1], dim=1))
        return torch.cat([self.out_h(x), self.out_v(x)], dim=1)