__author__ = 'max'

import math
from typing import Tuple, Dict
from overrides import overrides
import torch
import torch.nn as nn

from mae.modules.encoders.encoder_cores.encoder_core import EncoderCore
from mae.modules.networks.resnet import ResNet


class ResnetEncoderCoreBinaryImage28x28(EncoderCore):
    """
    Resnet Core for binary image of 28x28 resolution.
    See paper https://arxiv.org/abs/1611.02731 for details
    """

    def __init__(self, nz):
        super(ResnetEncoderCoreBinaryImage28x28, self).__init__()
        self.nz = nz
        self.nc = 1
        hidden_units = 512
        self.main = nn.Sequential(
            ResNet(self.nc, [32, 64, 64], [2, 2, 2]),
            nn.Conv2d(64, hidden_units, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_units),
            nn.ELU(),
        )
        self.linear = nn.Linear(hidden_units, 2 * nz, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        m = self.main[1]
        assert isinstance(m, nn.Conv2d)
        nn.init.xavier_normal_(m.weight)

        m = self.main[2]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input):
        output = self.main(input)
        output = self.linear(output.view(output.size()[:2]))
        return output.chunk(2, 1)

    @overrides
    def output_size(self) -> Tuple:
        return self.nz,

    @classmethod
    def from_params(cls, params: Dict) -> "ResnetEncoderCoreBinaryImage28x28":
        return ResnetEncoderCoreBinaryImage28x28(**params)


class ResnetEncoderCoreColorImage32x32(EncoderCore):
    """
    Resnet core for color image (RGB) of 32x32 resolution.
    output z shape = [z_channels, 8, 8]
    """

    def __init__(self, z_channels):
        super(ResnetEncoderCoreColorImage32x32, self).__init__()
        self.z_channels = z_channels
        self.H = 8
        self.W = 8
        self.nc = 3
        self.main = nn.Sequential(
            ResNet(self.nc, [48, 48], [1, 1]),
            # state [48, 32, 32]
            nn.Conv2d(48, 96, 3, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ELU(),
            # state [96, 16, 16]
            ResNet(96, [96, 96], [1, 1]),
            # state [96, 16, 16]
            nn.Conv2d(96, 96, 3, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ELU(),
            # state [96, 8, 8]
            ResNet(96, [96, 96, 96], [1, 1, 1]),
            # state [96, 8, 8]
            nn.Conv2d(96, 48, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(),
            # state [48, 8, 8]
            nn.Conv2d(48, 2 * self.z_channels, 1, 1, bias=False),
            # [2 * z_channels, 8, 8]
        )
        self.reset_parameters()

    def reset_parameters(self):
        m = self.main[1]
        assert isinstance(m, nn.Conv2d)
        nn.init.xavier_normal_(m.weight)

        m = self.main[5]
        assert isinstance(m, nn.Conv2d)
        nn.init.xavier_normal_(m.weight)

        m = self.main[9]
        assert isinstance(m, nn.Conv2d)
        nn.init.xavier_normal_(m.weight)

        m = self.main[12]
        assert isinstance(m, nn.Conv2d)
        nn.init.xavier_normal_(m.weight)

        m = self.main[2]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

        m = self.main[6]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

        m = self.main[10]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # [batch, 2 * z_channels, 8, 8]
        output = self.main(input)
        # [batch, z_channels, 8, 8]
        return output.chunk(2, 1)

    @overrides
    def output_size(self) -> Tuple:
        return self.z_channels, self.H, self.W

    @classmethod
    def from_params(cls, params: Dict) -> "ResnetEncoderCoreColorImage32x32":
        return ResnetEncoderCoreColorImage32x32(**params)


ResnetEncoderCoreBinaryImage28x28.register('resnet_binary_28x28')
ResnetEncoderCoreColorImage32x32.register('resnet_color_32x32')
