__author__ = 'max'

from typing import Tuple, Dict
from overrides import overrides
import torch.nn as nn
import torch.nn.functional as F

from mae.modules.encoders.encoder_cores.encoder_core import EncoderCore
from mae.modules.networks.dense import DenseNet
from mae.modules.networks.weight_norm import Conv2dWeightNorm


class DenseNetEncoderCoreColorImage32x32(EncoderCore):
    """
    Resnet core for color image (RGB) of 32x32 resolution.
    output z shape = [z_channels, 8, 8]
    """

    def __init__(self, z_channels):
        super(DenseNetEncoderCoreColorImage32x32, self).__init__()
        self.z_channels = z_channels
        self.H = 8
        self.W = 8
        self.nc = 3
        self.main = nn.Sequential(
            DenseNet(self.nc, 15, 3),
            # state [48, 32, 32]
            Conv2dWeightNorm(48, 48, 3, 2, 1, bias=True),
            nn.ELU(),
            # state [48, 16, 16]
            DenseNet(48, 16, 3),
            # state [96, 16, 16]
            Conv2dWeightNorm(96, 96, 3, 2, 1, bias=True),
            nn.ELU(),
            # state [96, 8, 8]
            DenseNet(96, 16, 6),
            # state [192, 8, 8]
            Conv2dWeightNorm(192, 96, 1, 1, bias=True),
            nn.ELU(),
            # state [96, 8, 8]
        )
        self.output = Conv2dWeightNorm(96, 2 * self.z_channels, 1, 1, bias=True)
        # [2 * z_channels, 8, 8]

    @overrides
    def initialize(self, x, init_scale=1.0):
        output = x
        for layer in self.main:
            if isinstance(layer, nn.ELU):
                output = layer(output)
            else:
                output = layer.initialize(output, init_scale=init_scale)
        output = self.output(output)
        # [batch, z_channels, 8, 8]
        mu, logvar = output.chunk(2, 1)
        return mu, F.hardtanh(logvar, min_val=-7, max_val=7.)

    def forward(self, input):
        # [batch, 192, 8, 8]
        output = self.main(input)
        # [batch, z_channels, 8, 8]
        output = self.output(output)
        mu, logvar = output.chunk(2, 1)
        return mu, F.hardtanh(logvar, min_val=-7, max_val=7.)

    @overrides
    def output_size(self) -> Tuple:
        return self.z_channels, self.H, self.W

    @classmethod
    def from_params(cls, params: Dict) -> "DenseNetEncoderCoreColorImage32x32":
        return DenseNetEncoderCoreColorImage32x32(**params)


DenseNetEncoderCoreColorImage32x32.register('densenet_color_32x32')
