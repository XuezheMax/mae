__author__ = 'max'

from collections import OrderedDict
import torch
import torch.nn as nn
from mae.modules.networks.weight_norm import Conv2dWeightNorm, ConvTranspose2dWeightNorm
from mae.modules.networks.resnet import conv3x3


__all__ = ['DenseNet', ]


class DenseNetBlock(nn.Module):
    def __init__(self, inplanes, growth_rate):
        super(DenseNetBlock, self).__init__()
        self.main = nn.Sequential(
            Conv2dWeightNorm(inplanes, 4 * growth_rate, kernel_size=1, bias=False),
            nn.ELU(),
            conv3x3(4 * growth_rate, growth_rate)
        )

    def forward(self, x):
        out = self.main(x)
        return torch.cat([x, out], dim=1)


class DenseNet(nn.Module):
    def __init__(self, inplanes, growth_rate, steps):
        super(DenseNet, self).__init__()
        blocks = []
        for step in range(steps):
            blocks.append(('block%d' % step, DenseNetBlock(inplanes, growth_rate)))
            blocks.append(('activation%d' % step, nn.ELU()))
            inplanes += growth_rate

        self.main = nn.Sequential(OrderedDict(blocks))

    def forward(self, x):
        return self.main(x)
