__author__ = 'max'

import math
from typing import Dict
import torch.nn as nn

from mae.modules.networks import DeResNet
from mae.modules.decoders.image_decoders.binary_image_decoder import BinaryImageDecoder


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


    @classmethod
    def from_params(cls, params: Dict) -> "ResnetDecoderBinaryImage28x28":
        return ResnetDecoderBinaryImage28x28(**params)


ResnetDecoderBinaryImage28x28.register('resnet_binary_28x28')
