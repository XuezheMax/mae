__author__ = 'max'

from overrides import overrides
import torch.nn as nn
import torch
from typing import Dict, Tuple, List

from mae.modules.flows.af.af import AF2d
from mae.modules.networks.auto_regressives.made import MADE2d


class AFMADEBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, num_hiddens, hidden_channels, hidden_kernels, order, bias=True, var=True):
        super(AFMADEBlock, self).__init__()
        self.mu = MADE2d(in_channels, kernel_size, num_hiddens, hidden_channels, hidden_kernels, order, bias=bias)
        if var:
            self.logvar = MADE2d(in_channels, kernel_size, num_hiddens, hidden_channels, hidden_kernels, order, bias=bias)
        else:
            self.logvar = None

    def forward(self, x):
        eps = 1e-12
        batch_size, in_channels, H, W = x.size()
        y = x.new_zeros(x.size())
        logstd = x.new_zeros(x.size())
        for _ in range(self._input_size):
            mu = self.mu(y)
            if self.logvar:
                logstd = self.logvar(y) * 0.5
            y = (x - mu).div(logstd.exp() + eps)
        return y, logstd.sum(dim=1)

    def backward(self, y):
        # [batch, x_shape]
        mu = self.mu(y)
        if self.logvar:
            logstd = self.logvar(y) * 0.5
        else:
            logstd = mu.new_zeros(mu.size())
        x = mu + y * logstd.exp()
        return x, logstd.sum(dim=1)

class AFMADE2d(AF2d):
    """

    """