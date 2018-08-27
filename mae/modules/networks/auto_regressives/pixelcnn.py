__author__ = 'max'

import math
import torch.nn as nn
from mae.modules.networks.masked import MaskedConv2d


class PixelCNNBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(PixelCNNBlock, self).__init__()
        self.mask_type = 'B'
        padding = kernel_size // 2
        out_channels = in_channels // 2

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            MaskedConv2d(self.mask_type, out_channels, out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.activation = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        m = self.main[0]
        assert isinstance(m, nn.Conv2d)
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        nn.init.normal_(m.weight, 0, math.sqrt(2. / n))

        m = self.main[6]
        assert isinstance(m, nn.Conv2d)
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        nn.init.normal_(m.weight, 0, math.sqrt(2. / n))

        m = self.main[1]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

        m = self.main[4]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

        m = self.main[7]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


    def forward(self, input):
        return self.activation(self.main(input) + input)


class MaskABlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, masked_channels):
        super(MaskABlock, self).__init__()
        self.mask_type = 'A'
        padding = kernel_size // 2

        self.main = nn.Sequential(
            MaskedConv2d(self.mask_type, masked_channels, in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        m = self.main[1]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.main(input)


class PixelCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, kernel_sizes, masked_channels):
        super(PixelCNN, self).__init__()
        assert num_blocks == len(kernel_sizes)
        self.blocks = []
        for i in range(num_blocks):
            if i == 0:
                block = MaskABlock(in_channels, out_channels, kernel_sizes[i], masked_channels)
            else:
                block = PixelCNNBlock(out_channels, kernel_sizes[i])
            self.blocks.append(block)

        self.main = nn.ModuleList(self.blocks)

        self.direct_connects = []
        for i in range(1, num_blocks - 1):
            self.direct_connects.append(PixelCNNBlock(out_channels, kernel_sizes[i]))

        self.direct_connects = nn.ModuleList(self.direct_connects)

    def forward(self, input):
        # [batch, out_channels, H, W]
        direct_inputs = []
        for i, layer in enumerate(self.main):
            if i > 2:
                direct_input = direct_inputs.pop(0)
                direct_conncet = self.direct_connects[i - 3]
                input = input + direct_conncet(direct_input)

            input = layer(input)
            direct_inputs.append(input)
        assert len(direct_inputs) == 3, 'architecture error: %d' % len(direct_inputs)
        direct_conncet = self.direct_connects[-1]
        return input + direct_conncet(direct_inputs.pop(0))
