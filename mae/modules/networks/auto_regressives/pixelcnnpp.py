__author__ = 'max'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mae.modules.networks.masked import DownShiftConv2d, DownRightShiftConv2d
from mae.modules.networks.masked import DownShiftConvTranspose2d, DownRightShiftConvTranspose2d


class GatedResnetBlock(nn.Module):
    def __init__(self, in_channels, h_channels=0, dropout=0.0):
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

        if h_channels:
            self.h_conv = nn.Conv2d(h_channels, in_channels, kernel_size=(3, 3), padding=1)
            self.h_bn = nn.BatchNorm2d(in_channels)
        else:
            self.h_conv = None
            self.h_bn = None

        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

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

        nn.init.constant_(self.h_bn.weight, 1)
        nn.init.constant_(self.h_bn.bias, 0)

    def forward(self, x1, x2, h=None):
        if h is not None:
            hc = self.h_bn(self.h_conv(h))
        else:
            hc = 0

        c1 = F.elu(self.down_bn1(self.down_conv1(x1)))
        # dropout
        c1 = self.dropout(c1)
        a1, b1 = (self.down_bn2(self.down_conv2(c1)) + hc).chunk(2, 1)
        c1 = F.elu(a1 * F.sigmoid(b1) + x1)

        c2 = self.down_right_bn1(self.down_right_conv1(x2))
        c2 = F.elu(c2 + self.nin_bn(self.nin(c1)))
        # dropout
        c2 = self.dropout(c2)
        a2, b2 = (self.down_right_bn2(self.down_right_conv2(c2)) + hc).chunk(2, 1)
        c2 = F.elu(a2 * F.sigmoid(b2) + x2)

        return c1, c2


class TopShitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TopShitBlock, self).__init__()
        self.down_conv1 = DownShiftConv2d(in_channels, out_channels, kernel_size=(2, 3), bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.down_conv2 = DownShiftConv2d(in_channels, out_channels, kernel_size=(1, 3), bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.down_right_conv = DownRightShiftConv2d(in_channels, out_channels, kernel_size=(2, 1), bias=True)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.reset_parameters()

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


class DownSamplingBlock(nn.Module):
    def __init__(self, num_filters):
        super(DownSamplingBlock, self).__init__()
        self.down_conv = DownShiftConv2d(num_filters, num_filters, kernel_size=(2, 3), stride=(2, 2), bias=True)
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.down_right_conv = DownRightShiftConv2d(num_filters, num_filters, kernel_size=(2, 2), stride=(2, 2), bias=True)
        self.bn2 = nn.BatchNorm2d(num_filters)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x1, x2, h=None):
        x1 = self.bn1(self.down_conv(x1))
        x2 = self.bn2(self.down_right_conv(x2))
        return F.elu(x1), F.elu(x2)


class UpSamplingBlock(nn.Module):
    def __init__(self, num_filters):
        super(UpSamplingBlock, self).__init__()
        self.down_deconv = DownShiftConvTranspose2d(num_filters, num_filters, kernel_size=(2, 3), stride=(2, 2), bias=True)
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.down_right_deconv = DownRightShiftConvTranspose2d(num_filters, num_filters, kernel_size=(2, 2), stride=(2, 2), bias=True)
        self.bn2 = nn.BatchNorm2d(num_filters)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

    def forward(self, x1, x2, h=None):
        x1 = self.bn1(self.down_deconv(x1))
        x2 = self.bn2(self.down_right_deconv(x2))
        return F.elu(x1), F.elu(x2)


class NINBlock(nn.Module):
    def __init__(self, num_filters):
        super(NINBlock, self).__init__()
        self.nin = nn.Conv2d(num_filters, num_filters, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(num_filters)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.nin.weight)

        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, residual):
        residual = self.bn(self.nin(residual))
        return F.elu(x + residual)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class PixelCNNPP(nn.Module):
    def __init__(self, levels, in_channels, out_channels, num_resnets, h_channels=0, dropout=0.0):
        super(PixelCNNPP, self).__init__()
        up_layers = []
        nins1 = []
        nins2 = []
        up_hs = []

        ori_h_channels = h_channels
        # ////////// up pass //////////
        for l in range(levels):
            if l == 0:
                up_layers.append(TopShitBlock(in_channels, out_channels))
                up_hs.append(Identity())
            else:
                up_layers.append(DownSamplingBlock(out_channels))
                nins1.append(NINBlock(out_channels))
                nins2.append(NINBlock(out_channels))
                if h_channels:
                    up_hs.append(nn.Conv2d(h_channels, h_channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=1))
                    h_channels = h_channels * 2
                else:
                    up_hs.append(Identity())

            for rep in range(num_resnets):
                up_layers.append(GatedResnetBlock(out_channels, h_channels, dropout))
                nins1.append(NINBlock(out_channels))
                nins2.append(NINBlock(out_channels))
                up_hs.append(Identity())

        # ////////// down pass //////////
        down_layers = []
        down_hs = []
        for l in range(levels):
            for rep in range(num_resnets):
                down_layers.append(GatedResnetBlock(out_channels, h_channels, dropout))
                down_hs.append(Identity())

            if l < levels - 1:
                down_layers.append(UpSamplingBlock(out_channels))
                if h_channels:
                    down_hs.append(nn.ConvTranspose2d(h_channels, h_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=1, output_padding=1))
                    h_channels = h_channels // 2
                else:
                    down_hs.append(Identity())

        assert len(nins1) == len(down_layers)
        self.up_layers = nn.ModuleList(up_layers)
        self.down_layers = nn.ModuleList(down_layers)
        self.nins1 = nn.ModuleList(nins1)
        self.nins2 = nn.ModuleList(nins2)

        assert ori_h_channels == h_channels
        assert len(up_hs) == len(up_layers)
        assert len(down_hs) == len(down_layers)
        self.up_hs = nn.ModuleList(up_hs)
        self.down_hs = nn.ModuleList(down_hs)

    def forward(self, input, h=None):
        up_pass = []
        x1 = None
        x2 = None
        for l, (layer, up_h) in enumerate(zip(self.up_layers, self.up_hs)):
            if l == 0:
                x1, x2 = layer(input)
            else:
                x1, x2 = layer(x1, x2, h=h)
            h = up_h(h)
            up_pass.append((x1, x2))

        assert len(up_pass) == len(self.nins1)

        for layer, down_h, nin1, nin2 in zip(self.down_layers, self.down_hs, self.nins1, self.nins2):
            u1, u2 = up_pass.pop()
            x1 = nin1(x1, u1)
            x2 = nin2(x2, u2)
            x1, x2 = layer(x1, x2, h)
            h = down_h(h)

        return x2
