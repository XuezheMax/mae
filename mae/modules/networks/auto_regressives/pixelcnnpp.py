__author__ = 'max'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mae.modules.networks.masked import DownShiftConv2d, DownRightShiftConv2d


class GatedResnetBlock(nn.Module):
    def __init__(self, in_channels, dropout):
        super(GatedResnetBlock, self).__init__()
        # Todo weight normalization vs batch normalization
        self.down_conv1 = DownShiftConv2d(in_channels, in_channels, kernel_size=(2, 3), bias=True)
        self.down_bn1 = nn.BatchNorm2d(in_channels)
        self.down_conv2 = DownShiftConv2d(in_channels, 2 * in_channels, kernel_size=(2, 3), bias=True)
        self.down_bn2 = nn.BatchNorm2d(2 * in_channels)

        self.down_right_conv1 = DownRightShiftConv2d(in_channels, in_channels, kernel_size=(2, 2), bias=True)
        self.down_right_bn1 = nn.BatchNorm2d(in_channels)
        self.nin = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.nin_bn = nn.BatchNorm2d(in_channels)
        self.down_right_conv2 = DownRightShiftConv2d(in_channels, 2 * in_channels, kernel_size=(2, 2), bias=True)
        self.down_right_bn2 = nn.BatchNorm2d(2 * in_channels)
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        nn.init.constant_(self.down_bn1.weight, 1)
        nn.init.constant_(self.down_bn1.bias, 0)

        nn.init.constant_(self.down_bn2.weight, 1)
        nn.init.constant_(self.down_bn2.bias, 0)

        nn.init.constant_(self.down_right_bn1.weight, 1)
        nn.init.constant_(self.down_right_bn1.bias, 0)

        nn.init.constant_(self.down_right_bn2.weight, 1)
        nn.init.constant_(self.down_right_bn2.bias, 0)

        nn.init.constant_(self.nin_bn.weight, 1)
        nn.init.constant_(self.nin_bn.bias, 0)

    def forward(self, x1, x2):
        c1 = F.elu(self.down_bn1(self.down_conv1(x1)))
        # dropout
        c1 = self.dropout(c1)
        a1, b1 = self.down_bn2(self.down_conv2(c1)).chunk(2, 1)
        c1 = a1 * F.sigmoid(b1)

        c2 = self.down_right_bn1(self.down_right_conv1(x2))
        c2 = F.elu(c2 + self.nin_bn(self.nin(x1)))
        # dropout
        c2 = self.dropout(c2)
        a2, b2 = self.down_right_bn2(self.down_right_conv2(c2)).chunk(2, 1)
        c2 = a2 * F.sigmoid(b2)

        return F.elu(c1 + x1), F.elu(c2 + x2)


class TopShitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TopShitBlock, self).__init__()
        self.down_conv1 = DownShiftConv2d(in_channels, out_channels, kernel_size=(2, 3), bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.down_conv2 = DownShiftConv2d(in_channels, out_channels, kernel_size=(1, 3), bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.down_right_conv = DownRightShiftConv2d(in_channels, out_channels, kernel_size=(2, 1), bias=True)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def reset_parameters(self):
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

        nn.init.constant_(self.bn3.weight, 1)
        nn.init.constant_(self.bn3.bias, 0)

    @staticmethod
    def down_shift(input):
        batch_size, channels, H, W = input.size()
        return torch.cat(input.new_zeros(batch_size, channels, 1, W), input[:, :, :H-1, :], dim=2)

    @staticmethod
    def right_shift(self, input):
        batch_size, channels, H, W = input.size()
        return torch.cat(input.new_zeros(batch_size, channels, H, 1), input[:, :, :, :W-1], dim=3)

    def forward(self, input):
        x1 = TopShitBlock.down_shift(self.bn1(self.down_conv1(input)))

        x2 = TopShitBlock.down_shift(self.bn2(self.down_conv2(input)))
        x2 = x2 + TopShitBlock.right_shift(self.bn3(self.down_right_conv(input)))

        return F.elu(x1), F.elu(x2)

class PixelCNNPP(nn.Module):
    pass
