import torch
import torch.nn as nn
from WSConv2d import WSConv2d
from PixelNorm import PixelNorm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixel_norm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky_RELU = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pixel_norm = use_pixel_norm

    def forward(self, X):
        X = self.leaky_RELU(self.conv1(X))
        X = self.pn(X) if self.use_pixel_norm else X
        X = self.leaky_RELU(self.conv2(X))
        X = self.pn(X) if self.use_pixel_norm else X
        return X

