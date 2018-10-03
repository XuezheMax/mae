__author__ = 'max'

import torch.nn as nn
from mae.modules.networks.masked import MaskedConv2d
from mae.modules.networks.weight_norm import Conv2dWeightNorm


class PixelCNNBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(PixelCNNBlock, self).__init__()
        self.mask_type = 'B'
        padding = kernel_size // 2
        out_channels = in_channels // 2

        self.main = nn.Sequential(
            Conv2dWeightNorm(in_channels, out_channels, 1, bias=True),
            nn.ELU(),
            MaskedConv2d(out_channels, out_channels, kernel_size, mask_type='B', padding=padding, bias=True),
            nn.ELU(),
            Conv2dWeightNorm(out_channels, in_channels, 1, bias=True),
        )
        self.activation = nn.ELU()

    def initialize(self, x, init_scale=1.0):
        out = x
        for layer in self.main:
            if isinstance(layer, nn.ELU):
                out = layer(out)
            else:
                out = layer.initialize(out, init_scale=init_scale)
        return self.activation(out + x)

    def forward(self, input):
        return self.activation(self.main(input) + input)


class MaskABlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, masked_channels):
        super(MaskABlock, self).__init__()
        self.mask_type = 'A'
        padding = kernel_size // 2

        self.main = nn.Sequential(
            MaskedConv2d(in_channels, out_channels, kernel_size, mask_type='A', masked_channels=masked_channels, padding=padding, bias=True),
            nn.ELU(),
        )

    def initialize(self, x, init_scale=1.0):
        assert len(self.main) == 2
        x = self.main[0].initialize(x, init_scale=init_scale)
        return self.main[1](x)

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

    def initialize(self, x, init_scale=1.0):
        # [batch, out_channels, H, W]
        direct_inputs = []
        for i, layer in enumerate(self.main):
            if i > 2:
                direct_input = direct_inputs.pop(0)
                direct_conncet = self.direct_connects[i - 3]
                x = x + direct_conncet.initialize(direct_input, init_scale=init_scale)

            x = layer.initialize(x, init_scale=init_scale)
            direct_inputs.append(x)
        assert len(direct_inputs) == 3, 'architecture error: %d' % len(direct_inputs)
        direct_conncet = self.direct_connects[-1]
        return x + direct_conncet.initialize(direct_inputs.pop(0), init_scale=init_scale)

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
