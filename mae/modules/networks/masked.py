__author__ = 'max'

import math
import numpy as np
import torch
import torch.nn as nn


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask_type, total_units, max_units=None, bias=True):
        '''
        Linear module with mask for auto-regressive models.
        Args:
            in_features: number of units in the inputs
            out_features: number of units in the outputs.
            max_units: the list containing the maximum units each input unit depends on.
            mask_type: type of the masked linear.
            total_units: the total number of units to assign.
            bias: using bias vector.
        '''
        super(MaskedLinear, self).__init__(in_features, out_features, bias=bias)
        layer_type, order = mask_type
        assert layer_type in {'input-hidden', 'hidden-hidden', 'hidden-output', 'input-output'}
        assert order in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())

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

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=0.1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x):
        self.weight.data.mul_(self.mask)
        return super(MaskedLinear, self).forward(x)


class MaskedConv2d(nn.Conv2d):
    """
    Conv2d with mask for pixelcnn.
    """
    def __init__(self, mask_type, masked_channels, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :masked_channels, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :masked_channels, kH // 2 + 1:] = 0

    def reset_parameters(self):
        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        nn.init.normal_(self.weight, 0, math.sqrt(2. / n))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        self.weight.data.mul_(self.mask)
        return super(MaskedConv2d, self).forward(x)