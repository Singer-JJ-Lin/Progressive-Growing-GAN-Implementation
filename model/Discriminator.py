import torch
import torch.nn as nn
from WSConv2d import WSConv2d
from ConvBlock import ConvBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super().__init__()
        self.factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
        self.leaky = nn.LeakyReLU(0.2)

        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.progressive_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()

        for i in range(len(self.factors) - 1, 0, -1):
            conv_in_channels = int(in_channels * self.factors[i])
            conv_out_channels = int(in_channels * self.factors[i-1])
            self.progressive_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pixel_norm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_channels, kernel_size=1, stride=1, padding=0))

        self.rgb_layers.append(self.initial_rgb)

        self.final_blocks = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )

    def fade_in(self, alpha, downscale, out):
        return alpha * out + (1 - alpha) * downscale

    def mini_batch_std(self, X):
        return torch.cat([X, torch.std(X, dim=0).mean().repeat(X.shape[0], 1, X.shape[2], X.shape[3])], dim=1)

    def forward(self, X, alpha, steps):
        cur_step = len(self.progressive_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](X))

        if steps == 0:
            out = self.mini_batch_std(out)
            return self.final_blocks(out).view(out.shape[0], -1)

        downscale = self.leaky(self.rgb_layers[cur_step+1](self.avg_pool(X)))
        out = self.avg_pool(self.progressive_blocks[cur_step](out))
        out = self.fade_in(alpha, downscale, out)

        for step in range(cur_step+1, len(self.progressive_blocks)):
            out = self.progressive_blocks[step](out)
            out = self.avg_pool(out)

        out = self.mini_batch_std(out)
        out = self.final_blocks(out).view(out.shape[0], -1)
        return out



