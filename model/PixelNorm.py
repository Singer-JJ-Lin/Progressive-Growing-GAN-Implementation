import torch
import torch.nn as nn


# 像素归一化
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, X):
        # dim=1表示对第1维做平均，keepdim保证维度不变
        return X / torch.sqrt(torch.mean(X ** 2, dim=1, keepdim=True) + self.epsilon)