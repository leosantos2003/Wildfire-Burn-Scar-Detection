# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    #

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class CBAM(nn.Module):
    """ Módulo de Atenção: Convolutional Block Attention Module (CBAM) """
    #
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, 1)
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(gate_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Atenção de Canal
        channel_att = torch.sigmoid(self.channel_gate(x))
        x = x * channel_att
        # Atenção Espacial
        spatial_att = self.spatial_gate(x)
        x = x * spatial_att
        return x

class AttentionUNetSiamese(nn.Module):
    def __init__(self, n_channels=9, n_classes=1):
        super(AttentionUNetSiamese, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- Encoder ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(128, 256)
        
        # --- Módulos de Atenção ---
        #
        self.cbam1 = CBAM(128)
        self.cbam2 = CBAM(256)

        # --- Decoder ---
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv1 = DoubleConv(256 + 128 + 128, 128) # Canal extra para a diferença
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv2 = DoubleConv(128 + 64 + 64, 64) # Canal extra para a diferença
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward_encoder(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)
        return x1, x2, x3

    def forward(self, t1, t2):
        # Processa as duas imagens pelo mesmo encoder
        x1_t1, x2_t1, x3_t1 = self.forward_encoder(t1)
        x1_t2, x2_t2, x3_t2 = self.forward_encoder(t2)

        # --- Fusão Bi-Temporal Avançada ---
        #
        
        # Nível 3 (bottleneck)
        x3_diff = torch.abs(x3_t1 - x3_t2)
        x3_fused = self.cbam2(x3_diff) # Aplica atenção na diferença

        # Nível 2
        x2_fused = self.cbam1(torch.abs(x2_t1 - x2_t2))

        # --- Decoder com Skip Connections e Fusão ---
        x = self.up1(x3_fused)
        x = torch.cat([x, x2_t1, x2_t2], dim=1) # Concatena com features originais
        x = self.dconv1(x)

        x = self.up2(x)
        x = torch.cat([x, x1_t1, x1_t2], dim=1)
        x = self.dconv2(x)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)