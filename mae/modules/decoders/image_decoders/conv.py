__author__ = 'max'

from typing import Dict
import torch.nn as nn
from mae.modules.decoders.image_decoders.binary_image_decoder import BinaryImageDecoder


class ConvDecoderBinaryImage28x28(BinaryImageDecoder):
    """
    Convolution Deocder for binary image of 28x28 resolution.
    See paper https://arxiv.org/abs/1511.06434 for details.
    """

    def __init__(self, nz, ngf, ngpu=1):
        super(ConvDecoderBinaryImage28x28, self).__init__(nz, ngpu=ngpu)
        self.ngf = ngf
        self.nc = 1
        self.core = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # # state size. (ngf) x 14 x 14
            nn.ConvTranspose2d(ngf, self.nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 28 x 28
        )
        self.reset_parameters()

        if ngpu > 1:
            self.core = nn.DataParallel(self.core, device_ids=list(range(ngpu)))

    def reset_parameters(self):
        for m in self.core.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0., 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def from_params(cls, params: Dict) -> "ConvDecoderBinaryImage28x28":
        return ConvDecoderBinaryImage28x28(**params)


ConvDecoderBinaryImage28x28.register('conv_binary_28x28')
