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
        self.main = nn.Sequential(
            ResNet(self.nc, [32, 64, 64], [2, 2, 2]),
            nn.Conv2d(64, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ELU(),
        )
        self.linear = nn.Linear(512, 2 * nz)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.main.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

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
    """

    def __init__(self):
        super(ResnetEncoderCoreColorImage32x32, self).__init__()
        self.nz_channels = 16
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
            nn.Tanh(),
            # state [48, 8, 8]
            nn.Conv2d(48, 2 * self.nz_channels, 1, 1, bias=False),
            # [2 * nz_channels, 8, 8]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.main.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # [batch, 2 * nz_channels, 8, 8]
        output = self.main(input)
        # [batch, nz_channels, 8, 8]
        return output.chunk(2, 1)

    @overrides
    def output_size(self) -> Tuple:
        return self.nz_channels, self.H, self.W

    @classmethod
    def from_params(cls, params: Dict) -> "ResnetEncoderCoreColorImage32x32":
        assert len(params) == 0
        return ResnetEncoderCoreColorImage32x32()


ResnetEncoderCoreBinaryImage28x28.register('resnet_binary_28x28')
ResnetEncoderCoreColorImage32x32.register('resnet_color_32x32')
