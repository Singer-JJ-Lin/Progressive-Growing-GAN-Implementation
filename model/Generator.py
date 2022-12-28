import torch
import torch.nn as nn
from WSConv2d import WSConv2d
from PixelNorm import PixelNorm
from ConvBlock import ConvBlock


class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        self.factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels),
            nn.LeakyReLU(0.2),
            PixelNorm())

        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1)
        self.progressive_block, self.rgb_layer = nn.ModuleList(), nn.ModuleList()
        self.rgb_layer.append(self.initial_rgb)

        for i in range(len(self.factors) - 1):
            conv_in_channels = int(in_channels * self.factors[i])
            conv_out_channels = int(in_channels * self.factors[i+1])
            self.progressive_block.append(ConvBlock(conv_in_channels, conv_out_channels))
            self.rgb_layer.append(WSConv2d(conv_out_channels, img_channels, kernel_size=1, stride=1))

    def fade_in(self, alpha, upscale, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscale)

    def forward(self, X, alpha, steps):
        out, upscale = self.initial(X), 0

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscale = torch.nn.functional.interpolate(input=out, scale_factor=2, mode="nearest")
            out = self.progressive_block[step](upscale)

        final_upscale = self.rgb_layer[steps-1](upscale)
        final_out = self.rgb_layer[steps](out)
        return self.fade_in(alpha, final_upscale, final_out)






