import torch as t
from torch import nn
import numpy as np


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.initial_layers = nn.Sequential(
                ConvLayer(3, 32, kernel_size = 9, stride = 1),
                nn.InstanceNorm2d(32, affine = True),
                nn.ReLU(True),
                ConvLayer(32, 64, kernel_size = 3, stride = 2),
                nn.InstanceNorm2d(64, affine = True),
                nn.ReLU(True),
                ConvLayer(64, 128, kernel_size = 3, stride = 2),
                nn.InstanceNorm2d(128, affine = True),
                nn.ReLU(True)
            )
        self.res_layers = nn.Sequential(
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128)
            )

        self.upsample_layers = nn.Sequential(
                UpsampleConvLayer(128, 64, kernel_size = 3, stride = 1, upsample = 2),
                nn.InstanceNorm2d(64, affine = True),
                nn.ReLU(True),
                UpsampleConvLayer(64, 32, kernel_size = 3, stride = 1, upsample = 2),
                nn.InstanceNorm2d(32, affine = True),
                nn.ReLU(True),
                ConvLayer(32, 3, kernel_size = 9, stride = 1)
            )
    def forward(self, x):
        #x = self.initial_layers(x)
        x = self.initial_layers(x)
        x = self.res_layers(x)
        x = self.upsample_layers(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size/2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample = None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = int(np.floor(kernel_size/2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = t.nn.functional.interpolate(x_in, scale_factor = self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size = 3, stride = 1)
        self.in1 = nn.InstanceNorm2d(channels, affine = True)
        self.conv2 = ConvLayer(channels, channels, kernel_size = 3, stride = 1)
        self.in2 = nn.InstanceNorm2d(channels, affine = True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out
