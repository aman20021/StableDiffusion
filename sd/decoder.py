import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VAE_ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.silu1 = nn.SiLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        
