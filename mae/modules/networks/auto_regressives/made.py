__author__ = 'max'

import torch.nn as nn
from mae.modules.networks.masked import MaskedLinear


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

        # weight normalization
        self.input_layer = nn.utils.weight_norm(self.input_layer)
        self.direct_connect = nn.utils.weight_norm(self.direct_connect)

        max_units = self.input_layer.max_units
        self.hidden_layers = []
        for hid in range(1, num_hiddens):
            hidden_layer = MaskedLinear(hidden_size, hidden_size, ('hidden-hidden', order), total_units, max_units=max_units, bias=bias)
            max_units = hidden_layer.max_units
            # weight normalization
            hidden_layer = nn.utils.weight_norm(hidden_layer)
            self.hidden_layers.append(hidden_layer)
            self.add_module('hidden%d' % hid, hidden_layer)
        assert self.num_hiddens == len(self.hidden_layers) + 1

        self.output_layer = MaskedLinear(hidden_size, input_size, ('hidden-output', order), total_units, max_units=max_units, bias=bias)
        # weight normalization
        self.output_layer = nn.utils.weight_norm(self.output_layer)

    def forward(self, x):
        output = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            output = self.activation(hidden_layer(output))
        return self.output_layer(output) + self.direct_connect(x)
