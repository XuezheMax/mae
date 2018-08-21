__author__ = 'max'

from typing import Tuple, Dict
from overrides import overrides
import torch
import torch.nn as nn

from mae.modules.encoders.encoder_cores.encoder_core import EncoderCore


class ConvEncoderCoreBinaryImage28x28(EncoderCore):
    """
    Convolution Core for binary image of 28x28 resolution.
    See paper https://arxiv.org/abs/1511.06434 for details.
    """

    def __init__(self, nz, ndf):
        super(ConvEncoderCoreBinaryImage28x28, self).__init__()
        self.nz = nz
        self.ndf = ndf
        self.nc = 1
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(self.nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 2 * nz, 4, 1, 0, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for m in self.main.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.normal_(0.0, 0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.normal_(1.0, 0.02)
                    m.bias.zero_()

    def forward(self, input):
        output = self.main(input)
        return output.view(output.size()[:2]).chunk(2, 1)

    @overrides
    def output_size(self) -> Tuple:
        return self.nz,

    @classmethod
    def from_params(cls, params: Dict) -> "ConvEncoderCoreBinaryImage28x28":
        return ConvEncoderCoreBinaryImage28x28(**params)


ConvEncoderCoreBinaryImage28x28.register('conv_binary_28x28')
