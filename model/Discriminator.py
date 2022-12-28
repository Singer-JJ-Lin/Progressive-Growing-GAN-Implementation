import torch
import torch.nn as nn
from WSConv2d import WSConv2d
from ConvBlock import ConvBlock

class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels):
        super().__init__()
        self.factors = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 1, 1, 1]
        self.leaky = nn.LeakyReLU(0.2)
        self.initial_rgb = WSConv2d
        self.progressive_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()

        for i in range(len(self.factors) - 1):
            conv_in_channels = int(in_channels * self.factors[i])
            conv_out_channels = int(in_channels * self.factors[i+1])
            self.progressive_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pixel_norm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_channels, kernel_size=1, stride=1))


    def fade_in(self, alpha, upscale, generated):
        return alpha * generated + (1 - alpha) * upscale

    def mini_batch_std(self, X):
        pass

    def forward(self, X, alpha, steps):
        pass