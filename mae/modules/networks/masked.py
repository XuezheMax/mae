__author__ = 'max'

import math
from overrides import overrides
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.utils import _pair

from mae.modules.utils import norm


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask_type, total_units, max_units=None, bias=True):
        """
        Linear module with mask for auto-regressive models.
        Args:
            in_features: number of units in the inputs
            out_features: number of units in the outputs.
            max_units: the list containing the maximum units each input unit depends on.
            mask_type: type of the masked linear.
            total_units: the total number of units to assign.
            bias: using bias vector.
        """
        super(MaskedLinear, self).__init__(in_features, out_features, bias=bias)
        layer_type, order = mask_type
        assert layer_type in {'input-hidden', 'hidden-hidden', 'hidden-output', 'input-output'}
        assert order in {'A', 'B'}

        # override the max_units for input layer
        if layer_type.startswith('input'):
            max_units = np.arange(in_features) + 1
        else:
            assert max_units is not None and len(max_units) == in_features

        if layer_type.endswith('output'):
            assert out_features > total_units
            self.max_units = np.arange(out_features)
            self.max_units[total_units:] = total_units
        else:
            units_per_units = float(total_units) / out_features
            self.max_units = np.zeros(out_features, dtype=np.int32)
            for i in range(out_features):
                self.max_units[i] = np.ceil((i + 1) * units_per_units)

        mask = np.zeros([out_features, in_features], dtype=np.float32)
        for i in range(out_features):
            for j in range(in_features):
                mask[i, j] = float(self.max_units[i] >= max_units[j])

        # reverse order
        if order == 'B':
            reverse_mask = mask[::-1, :]
            reverse_mask = reverse_mask[:, ::-1]
            mask = np.copy(reverse_mask)

        self.register_buffer('mask', torch.from_numpy(mask).float())
        self.reset_parameters()

    @overrides
    def reset_parameters(self):
        nn.init.xavier_uniform(self.weight, gain=0.1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    @overrides
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, type={}, order={}'.format(
            self.in_features, self.out_features, self.bias is not None,
            self.layer_type, self.order
        )


class MaskedLinearWeightNorm(nn.Module):
    def __init__(self, in_features, out_features, mask_type, total_units, max_units=None, bias=True):
        """
        Args:
            in_features: number of units in the inputs
            out_features: number of units in the outputs.
            max_units: the list containing the maximum units each input unit depends on.
            mask_type: type of the masked linear.
            total_units: the total number of units to assign.
            bias: using bias vector.
        """
        super(MaskedLinearWeightNorm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_v = Parameter(torch.Tensor(out_features, in_features))
        self.weight_g = Parameter(torch.Tensor(out_features, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        layer_type, order = mask_type
        self.layer_type = layer_type
        self.order = order
        assert layer_type in {'input-hidden', 'hidden-hidden', 'hidden-output', 'input-output'}
        assert order in {'A', 'B'}
        self.register_buffer('mask', self.weight_v.data.clone())

        # override the max_units for input layer
        if layer_type.startswith('input'):
            max_units = np.arange(in_features) + 1
        else:
            assert max_units is not None and len(max_units) == in_features

        if layer_type.endswith('output'):
            assert out_features > total_units
            self.max_units = np.arange(out_features)
            self.max_units[total_units:] = total_units
        else:
            units_per_units = float(total_units) / out_features
            self.max_units = np.zeros(out_features, dtype=np.int32)
            for i in range(out_features):
                self.max_units[i] = np.ceil((i + 1) * units_per_units)

        mask = np.zeros([out_features, in_features], dtype=np.float32)
        for i in range(out_features):
            for j in range(in_features):
                mask[i, j] = float(self.max_units[i] >= max_units[j])

        # reverse order
        if order == 'B':
            reverse_mask = mask[::-1, :]
            reverse_mask = reverse_mask[:, ::-1]
            mask = np.copy(reverse_mask)

        self.mask.copy_(torch.from_numpy(mask).float())
        self.reset_parameters()

        self._init = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_v, gain=0.1)
        self.weight_v.data.mul_(self.mask)
        _norm = norm(self.weight_v, 0).data + 1e-8
        self.weight_g.data.copy_(_norm.log())
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, input):
        self.weight_v.data.mul_(self.mask)
        _norm = norm(self.weight_v, 0) + 1e-8
        weight = self.weight_v * (self.weight_g.exp() / _norm)
        return F.linear(input, weight, self.bias)

    @overrides
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, type={}, order={}'.format(
            self.in_features, self.out_features, self.bias is not None,
            self.layer_type, self.order
        )


class MaskedConv2d(nn.Conv2d):
    """
    Conv2d with mask for pixelcnn.
    """

    def __init__(self, mask_type, masked_channels, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', torch.ones(self.weight.size()))
        _, _, kH, kW = self.weight.size()
        self.mask[:, :masked_channels, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :masked_channels, kH // 2 + 1:] = 0
        self.reset_parameters()

    @overrides
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class MaskedConv2dwithWeightNorm(nn.Module):
    """
    Conv2d with mask and weight normalization for MADE2d.
    """

    def __init__(self, mask_type, order, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2dwithWeightNorm, self).__init__()
        assert mask_type in {'A', 'B'}
        assert order in {'A', 'B'}
        self.mask_type = mask_type
        self.order = order
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight_v = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.weight_g = Parameter(torch.Tensor(out_channels, 1, 1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('mask', torch.ones(self.weight_v.size()))
        _, _, kH, kW = self.weight_v.size()
        mask = np.ones([*self.mask.size()], dtype=np.float32)
        mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        mask[:, :, kH // 2 + 1:] = 0

        # reverse order
        if order == 'B':
            reverse_mask = mask[:, :, ::-1, :]
            reverse_mask = reverse_mask[:, :, :, ::-1]
            mask = reverse_mask.copy()
        self.mask.copy_(torch.from_numpy(mask).float())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_v)
        self.weight_v.data.mul_(self.mask)
        _norm = norm(self.weight_v, 0).data + 1e-8
        self.weight_g.data.copy_(_norm.log())
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        self.weight_v.data.mul_(self.mask)
        _norm = norm(self.weight_v, 0) + 1e-8
        weight = self.weight_v * (self.weight_g.exp() / _norm)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ', type={mask_type}, order={order}'
        return s.format(**self.__dict__)


class DownShiftConv2d(nn.Conv2d):
    """
    Conv2d with down shift operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super(DownShiftConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, bias=bias)
        self.shift_padding = ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2, kernel_size[0] - 1, 0)
        self.reset_parameters()

    @overrides
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        input = F.pad(input, self.shift_padding)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class DownRightShiftConv2d(DownShiftConv2d):
    """
    Conv2d with dwon right shift operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super(DownRightShiftConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, bias=bias)
        self.shift_padding = (kernel_size[1] - 1, 0, kernel_size[0] - 1, 0)
        self.reset_parameters()


class DownShiftConvTranspose2d(nn.ConvTranspose2d):
    """
    ConvTranspose2d with down shift operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), bias=True):
        assert len(stride) == 2
        output_padding = (stride[0] - 1, stride[1] - 1)
        super(DownShiftConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, output_padding=output_padding, bias=bias)
        self.reset_parameters()

    @overrides
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        output_padding = self._output_padding(input, output_size=None)
        output = F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        size = output.size()
        H_padding = self.kernel_size[0] - 1
        W_padding = (self.kernel_size[1] - 1) // 2
        return output[:, :, :(size[2] - H_padding), W_padding:(size[3] - W_padding)]


class DownRightShiftConvTranspose2d(DownShiftConvTranspose2d):
    """
    ConvTranspose2d with down shift operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), bias=True):
        super(DownRightShiftConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, input):
        output_padding = self._output_padding(input, output_size=None)
        output = F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        size = output.size()
        H_padding = self.kernel_size[0] - 1
        W_padding = self.kernel_size[1] - 1
        return output[:, :, :(size[2] - H_padding),:(size[3] - W_padding)]
