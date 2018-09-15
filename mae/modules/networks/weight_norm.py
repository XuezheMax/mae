__author__ = 'max'

from overrides import overrides
import torch.nn as nn


class LinearWeightNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWeightNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
        self.linear = nn.utils.weight_norm(self.linear)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def forward(self, input):
        return self.linear(input)


class Conv2dWeightNorm(nn.Module):
    """
    Conv2d with weight normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWeightNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, input):
        return self.conv(input)

    @overrides
    def extra_repr(self):
        return self.conv.extra_repr()


class ConvTranspose2dWeightNorm(nn.Module):
    """
    Convolution transpose 2d with weight normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ConvTranspose2dWeightNorm, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                         padding=padding, output_padding=output_padding, groups=groups,
                                         bias=bias, dilation=dilation)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.deconv.weight)
        if self.deconv.bias is not None:
            nn.init.constant_(self.deconv.bias, 0)
        self.deconv = nn.utils.weight_norm(self.deconv)

    def _output_padding(self, input, output_size):
        return self.deconv._output_padding(input, output_size)

    def forward(self, input):
        return self.deconv(input)

    @overrides
    def extra_repr(self):
        return self.deconv.extra_repr()
