__author__ = 'max'

import math
from overrides import overrides
from typing import Dict, Tuple
import torch.nn as nn

from mae.modules.networks import DeResNet
from mae.modules.decoders.image_decoders.binary_image_decoder import BinaryImageDecoder
from mae.modules.decoders.image_decoders.color_image_decoder import ColorImageDecoder


class ResnetDecoderBinaryImage28x28(BinaryImageDecoder):
    """
    Resnet Deocder for binary image of 28x28 resolution.
    See paper https://arxiv.org/abs/1611.02731 for details
    """

    def __init__(self, nz, ngpu=1):
        super(ResnetDecoderBinaryImage28x28, self).__init__(nz, ngpu=ngpu)
        self.nc = 1
        self.core = nn.Sequential(
            nn.ConvTranspose2d(nz, 64, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            DeResNet(64, [64, 32, self.nc], [2, 2, 2], [0, 1, 1]),
            nn.Sigmoid(),
        )
        self.reset_parameters()

        if ngpu > 1:
            self.core = nn.DataParallel(self.core, device_ids=list(range(ngpu)))

    def reset_parameters(self):
        m = self.core[0]
        assert isinstance(m, nn.ConvTranspose2d)
        nn.init.xavier_normal_(m.weight)

        m = self.core[1]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    def forward(self, z):
        return self.core(z)

    @overrides
    def output_size(self) -> Tuple:
        """

        Returns: a tuple of the output shape of decoded x (excluding batch_size)

        """
        return 1, 28, 28

    @classmethod
    def from_params(cls, params: Dict) -> "ResnetDecoderBinaryImage28x28":
        return ResnetDecoderBinaryImage28x28(**params)


class ResnetDecoderColorImage32x32(ColorImageDecoder):
    """
    Resnet Deocder for color RGB image of 32x32x3 resolution.
    See paper https://arxiv.org/abs/1611.02731 for details
    """

    def __init__(self, z_channels, nmix, ngpu=1):
        super(ResnetDecoderColorImage32x32, self).__init__(nmix, ngpu=ngpu)
        self.z_channels = z_channels
        self.H = 8
        self.W = 8

        self.core = nn.Sequential(
            nn.ConvTranspose2d(self.z_channels, 96, 1, 1, 0, bias=False),
            nn.BatchNorm2d(96),
            nn.ELU(),
            # state [b, 96, 8, 8]
            DeResNet(96, [96, 96, 96], [1, 1, 1], [0, 0, 0]),
            # state [96, 8, 8]
            nn.ConvTranspose2d(96, 96, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ELU(),
            # state [96, 16, 16]
            DeResNet(96, [96, 96], [1, 1], [0, 0]),
            # state [96, 16, 16]
            nn.ConvTranspose2d(96, 48, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(),
            # state [48, 32, 32]
            DeResNet(48, [48, 48], [1, 1], [0, 0]),
            # state [48, 32, 32]
            nn.Conv2d(48, (self.nc * 3 + 1) * self.nmix, 1, bias=False)
            # state [(nc * 3 + 1) * nmix, 32, 32]
        )
        self.reset_parameters()

        if ngpu > 1:
            self.core = nn.DataParallel(self.core, device_ids=list(range(ngpu)))

    def reset_parameters(self):
        m = self.core[0]
        assert isinstance(m, nn.ConvTranspose2d)
        nn.init.xavier_normal_(m.weight)

        m = self.core[4]
        assert isinstance(m, nn.ConvTranspose2d)
        nn.init.xavier_normal_(m.weight)

        m = self.core[8]
        assert isinstance(m, nn.ConvTranspose2d)
        nn.init.xavier_normal_(m.weight)

        m = self.core[12]
        assert isinstance(m, nn.Conv2d)
        nn.init.xavier_normal_(m.weight)

        m = self.core[1]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

        m = self.core[5]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

        m = self.core[9]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    @overrides
    def z_shape(self) -> Tuple:
        return self.z_channels, self.H, self.W

    @overrides
    def output_size(self) -> Tuple:
        """

        Returns: a tuple of the output shape of decoded x (excluding batch_size)

        """
        return self.nc, 32, 32

    def forward(self, x, z):
        return self.core(z)

    @classmethod
    def from_params(cls, params: Dict) -> "ResnetDecoderColorImage32x32":
        return ResnetDecoderColorImage32x32(**params)


ResnetDecoderBinaryImage28x28.register('resnet_binary_28x28')
ResnetDecoderColorImage32x32.register('resnet_color_32x32')
