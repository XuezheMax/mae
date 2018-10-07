__author__ = 'max'

from collections import OrderedDict
import torch
import torch.nn as nn
from mae.modules.networks.weight_norm import Conv2dWeightNorm
from mae.modules.networks.resnet import conv3x3


__all__ = ['DenseNet', ]


class DenseNetBlock(nn.Module):
    def __init__(self, inplanes, growth_rate):
        super(DenseNetBlock, self).__init__()
        self.main = nn.Sequential(
            Conv2dWeightNorm(inplanes, 4 * growth_rate, kernel_size=1, bias=True),
            nn.ELU(),
            conv3x3(4 * growth_rate, growth_rate)
        )

    def initialize(self, x, init_scale=1.0):
        assert len(self.main) == 3
        out = self.main[0].initialize(x, init_scale=init_scale)
        out = self.main[1](out)
        out = self.main[2].initialize(out, init_scale=init_scale)
        return torch.cat([x, out], dim=1)

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

    def initialize(self, x, init_scale=1.0):
        for i, block in enumerate(self.main):
            if i % 2 == 0:
                x = block.initialize(x, init_scale=init_scale)
            else:
                x = block(x)
        return x

    def forward(self, x):
        return self.main(x)
