__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import torch.nn as nn

from mae.modules.networks import DeResNet, ConvTranspose2dWeightNorm, Conv2dWeightNorm
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
            ConvTranspose2dWeightNorm(nz, 64, 4, 1, 0, bias=True),
            nn.ELU(),
            DeResNet(64, [64, 32, self.nc], [2, 2, 2], [0, 1, 1]),
            nn.Sigmoid(),
        )

        if ngpu > 1:
            self.core = nn.DataParallel(self.core, device_ids=list(range(ngpu)))

    def forward(self, z):
        return self.core(z)

    @overrides
    def initialize(self, x, z, init_scale=1.0):
        core = self.core.module if isinstance(self.core, nn.DataParallel) else self.core
        assert len(core) == 4
        out = core[0].initialize(z, init_scale=init_scale)
        out = core[1](out)
        out = core[2].initialize(out, init_scale=init_scale)
        return core[3](out)

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
            ConvTranspose2dWeightNorm(self.z_channels, 96, 1, 1, 0, bias=True),
            nn.ELU(),
            # state [b, 96, 8, 8]
            DeResNet(96, [96, 96, 96], [1, 1, 1], [0, 0, 0]),
            # state [96, 8, 8]
            ConvTranspose2dWeightNorm(96, 96, 3, 2, 1, 1, bias=True),
            nn.ELU(),
            # state [96, 16, 16]
            DeResNet(96, [96, 96], [1, 1], [0, 0]),
            # state [96, 16, 16]
            ConvTranspose2dWeightNorm(96, 48, 3, 2, 1, 1, bias=True),
            nn.ELU(),
            # state [48, 32, 32]
            DeResNet(48, [48, 48], [1, 1], [0, 0]),
            # state [48, 32, 32]
            Conv2dWeightNorm(48, (self.nc * 3 + 1) * self.nmix, 1, bias=True)
            # state [(nc * 3 + 1) * nmix, 32, 32]
        )

        if ngpu > 1:
            self.core = nn.DataParallel(self.core, device_ids=list(range(ngpu)))

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

    @overrides
    def initialize(self, x, z, init_scale=1.0):
        core = self.core.module if isinstance(self.core, nn.DataParallel) else self.core
        output = z
        for layer in core:
            if isinstance(layer, nn.ELU):
                output = layer(output)
            else:
                output = layer.initialize(output, init_scale=init_scale)
        return output

    @classmethod
    def from_params(cls, params: Dict) -> "ResnetDecoderColorImage32x32":
        return ResnetDecoderColorImage32x32(**params)


ResnetDecoderBinaryImage28x28.register('resnet_binary_28x28')
ResnetDecoderColorImage32x32.register('resnet_color_32x32')
