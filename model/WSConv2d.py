import torch.nn as nn


#  实现了均衡学习率的卷积层
class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5

        # 这里复制了一份卷积层的偏置值
        self.bias = self.conv.bias
        self.conv.bias = None

        # 对卷积层进行初始化
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, X):
        # 这里对偏置值进行了reshape，使之可以与前面的相加
        return self.conv(X * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)







