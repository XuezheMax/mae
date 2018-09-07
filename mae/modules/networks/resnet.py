__author__ = 'max'

import math
import torch
import torch.nn as nn

__all__ = ['ResNet', 'DeResNet', ]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride, padding=1, bias=False)


def deconv3x3(in_planes, out_planes, stride=1, output_padding=0):
    "3x3 deconvolution with padding"
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1,
                              output_padding=output_padding, bias=False)


class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation = nn.ELU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        self.downsample = downsample
        self.stride = stride
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        residual = x if self.downsample is None else self.downsample(x)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.activation(out + residual)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        return out


class DeResNetBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, output_padding=0):
        super(DeResNetBlock, self).__init__()
        self.deconv1 = deconv3x3(inplanes, planes, stride, output_padding)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation = nn.ELU()
        self.deconv2 = deconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes,
                                   kernel_size=1, stride=stride,
                                   output_padding=output_padding, bias=False),
                nn.BatchNorm2d(planes),
            )
        self.downsample = downsample
        self.stride = stride
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                # m.weight.normal_(0, math.sqrt(2. / n))
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.activation(x)

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        residual = x if self.downsample is None else self.downsample(x)

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        out = self.deconv2(out)
        out = self.bn2(out)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        return out + residual


class ResNet(nn.Module):
    def __init__(self, inplanes, planes, strides):
        super(ResNet, self).__init__()
        assert len(planes) == len(strides)

        blocks = []
        for i in range(len(planes)):
            plane = planes[i]
            stride = strides[i]
            block = ResNetBlock(inplanes, plane, stride=stride)
            blocks.append(block)
            inplanes = plane

        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        return self.main(x)


class DeResNet(nn.Module):
    def __init__(self, inplanes, planes, strides, output_paddings):
        super(DeResNet, self).__init__()
        assert len(planes) == len(strides)
        assert len(planes) == len(output_paddings)

        blocks = []
        for i in range(len(planes)):
            plane = planes[i]
            stride = strides[i]
            output_padding = output_paddings[i]
            block = DeResNetBlock(inplanes, plane, stride=stride, output_padding=output_padding)
            blocks.append(block)
            inplanes = plane

        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        return self.main(x)
