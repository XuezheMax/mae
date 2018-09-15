__author__ = 'max'

import torch.nn as nn
from torch.nn.modules.utils import _pair
from mae.modules.networks.masked import MaskedLinear, MaskedConv2d


class MADE(nn.Module):
    """
    The MADE model. See https://arxiv.org/abs/1502.03509 for details.
    """

    def __init__(self, input_size, num_hiddens, hidden_size, order, bias=True):
        '''

        Args:
            input_size: number of input units
            num_hiddens: number of hidden layers
            hidden_size: the number of units in each hidden layer
            order: mask type (should be in {'A', 'B'})
            bias: using bias
            weight_norm: using weight normalization
        '''
        super(MADE, self).__init__()
        self.input_size = input_size
        self.num_hiddens = num_hiddens
        self.hidden_size = hidden_size
        self.activation = nn.ReLU()  # TODO other activation functions

        assert num_hiddens > 0
        total_units = input_size - 1

        self.input_layer = MaskedLinear(input_size, hidden_size, ('input-hidden', order), total_units, bias=bias)
        self.direct_connect = MaskedLinear(input_size, input_size, ('input-output', order), total_units, bias=False)

        max_units = self.input_layer.max_units
        self.hidden_layers = []
        for hid in range(1, num_hiddens):
            hidden_layer = MaskedLinear(hidden_size, hidden_size, ('hidden-hidden', order), total_units, max_units=max_units, bias=bias)
            max_units = hidden_layer.max_units
            self.hidden_layers.append(hidden_layer)
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        assert self.num_hiddens == len(self.hidden_layers) + 1

        self.output_layer = MaskedLinear(hidden_size, input_size, ('hidden-output', order), total_units, max_units=max_units, bias=bias)

    def forward(self, x):
        output = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            output = self.activation(hidden_layer(output))
        return self.output_layer(output) + self.direct_connect(x)


class MADE2d(nn.Module):
    """
    The MADE2d model.
    """

    def __init__(self, in_channels, kernel_size, num_hiddens, hidden_channels, hidden_kernels, order, bias=True):
        """

        Args:
            in_channels: int
                number of channels
            kernel_size: int or tuple
                kernel size
            num_hiddens: int
                number of hidden layers
            hidden_channels: int
                number of channels of hidden layers
            hidden_kernels: list of int or tuples
                kernel_size of each hidden layer
            order: 'A' or 'B'
            bias: using bias (default=True)
        """
        super(MADE2d, self).__init__()
        self.activation = nn.ELU() # TODO other activation functions
        assert num_hiddens > 0
        assert num_hiddens == len(hidden_kernels)
        gain=0.05
        kH, kW = _pair(kernel_size)
        padding = (kH // 2, kW // 2)
        self.top_layer = MaskedConv2d(in_channels, hidden_channels, kernel_size, mask_type='A', order=order, padding=padding, bias=bias, init_gain=gain)
        self.direct_connect = MaskedConv2d(in_channels, in_channels, kernel_size, mask_type='A', order=order, padding=padding, bias=False, init_gain=gain)

        self.hidden_layers = []
        for i in range(num_hiddens - 1):
            hidden_kernel = _pair(hidden_kernels[i])
            padding = (hidden_kernel[0] // 2, hidden_kernel[1] // 2)
            hidden_layer = MaskedConv2d(hidden_channels, hidden_channels, hidden_kernel, mask_type='B', order=order, padding=padding, bias=bias, init_gain=gain)
            self.hidden_layers.append(hidden_layer)
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        assert num_hiddens == len(self.hidden_layers) + 1
        out_kernel = _pair(hidden_kernels[-1])
        padding = (out_kernel[0] // 2, out_kernel[1] // 2)
        self.output_layer = MaskedConv2d(hidden_channels, in_channels, out_kernel, mask_type='B', order=order, padding=padding, bias=bias, init_gain=gain)

    def forward(self, x):
        output = self.activation(self.top_layer(x))
        for hidden_layer in self.hidden_layers:
            output = self.activation(hidden_layer(output) + output)
        return self.output_layer(output) + self.direct_connect(x)
