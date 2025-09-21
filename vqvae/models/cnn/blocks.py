import torch
import torch.nn as nn
import torch.nn.functional as F


def Normalize(num_channels, num_groups):
    """
    @func: 
    normalize each group along with the feature dimension | num_fea_each_goup = num_groups / num_channels

    """
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True) # group normalization


class Upsample2x_TF(nn.Module):
    """
    @func: 
    upsample | *2

    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
    def forward(self, x):
        return self.conv_transpose(x)


class Downsample2x(nn.Module):
    """
    @func: 
    downsample | /2

    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2) # /2
    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1), mode='constant', value=0.)) # F.pad: [left,right,top,bottom]


class ConvBlock(nn.Module):
    def __init__(self, *, in_channels, num_groups, out_channels=None, dropout=None):
        super().__init__()
        # init
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        # layer1
        self.norm1 = Normalize(num_channels=in_channels, num_groups=num_groups)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1) # only on time-dimension
        # layer2
        self.norm2 = Normalize(num_channels=out_channels, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity() # 
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) # only on time-dimension
        # residual
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()
    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True)) # F.silu is the Sigmoid Linear Unit 
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True))) # F.silu is the Sigmoid Linear Unit 
        return self.nin_shortcut(x) + h
