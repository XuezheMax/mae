__author__ = 'max'

from overrides import overrides
import torch.nn as nn
import torch
from typing import Dict, Tuple, List

from mae.modules.networks.auto_regressives.made import MADE
from mae.modules.flows.af.af import AF


class AFMADEBlock(nn.Module):
    def __init__(self, input_size, num_hiddens, hidden_size, order, bias=True, var=True, weight_norm=True):
        super(AFMADEBlock, self).__init__()
        self.mu = MADE(input_size, num_hiddens, hidden_size, order, bias=bias, weight_norm=weight_norm)
        if var:
            self.logvar = MADE(input_size, num_hiddens, hidden_size, order, bias=bias, weight_norm=weight_norm)
        else:
            self.logvar = None
        assert input_size > 0, 'input size (%s) should be positive' % input_size
        self._input_size = input_size

    def forward(self, x):
        eps = 1e-12
        y = x.new_zeros(x.size())
        logstd = x.new_zeros(x.size())
        for _ in range(self._input_size):
            mu = self.mu(y)
            if self.logvar:
                logstd = self.logvar(y) * 0.5
            y = (x - mu).div(logstd.exp() + eps)
        return y, logstd.sum(dim=1)

    def backward(self, y):
        # [batch, nz]
        mu = self.mu(y)
        if self.logvar:
            logstd = self.logvar(y) * 0.5
        else:
            logstd = mu.new_zeros(mu.size())
        x = mu + y * logstd.exp()
        return x, logstd.sum(dim=1)


class AFMADEDualBlock(nn.Module):
    def __init__(self, input_size, num_hiddens, hidden_size, bias=True, var=True, weight_norm=True):
        super(AFMADEDualBlock, self).__init__()
        self.fwd = AFMADEBlock(input_size, num_hiddens, hidden_size, 'A', bias=bias, var=var, weight_norm=weight_norm)
        self.bwd = AFMADEBlock(input_size, num_hiddens, hidden_size, 'B', bias=bias, var=var, weight_norm=weight_norm)

    def forward(self, x):
        # forward
        x, logdet_fwd = self.fwd.forward(x)
        y, logdet_bwd = self.bwd.forward(x)
        return y, logdet_fwd + logdet_bwd

    def backward(self, y):
        # backward
        # [batch, nz]
        y, logdet_bwd = self.bwd.backward(y)
        # [batch, nz]
        x, logdet_fwd = self.fwd.backward(y)
        return x, logdet_fwd + logdet_bwd


class AFMADE(AF):
    def __init__(self, input_size, num_blocks, num_hiddens=1, hidden_size=None, bias=True, var=True, weight_norm=True):
        super(AFMADE, self).__init__(input_size)
        self.num_blocks = num_blocks
        self.num_hiddens = num_hiddens
        hidden_size = input_size * 10 if hidden_size is None else hidden_size
        self.hidden_size = hidden_size
        self.blocks = []
        for i in range(num_blocks):
            block = AFMADEDualBlock(self.nz, num_hiddens, hidden_size, bias=bias, var=var, weight_norm=weight_norm)
            self.blocks.append(block)
        assert num_blocks == len(self.blocks)
        self.blocks = nn.ModuleList(self.blocks)

    @overrides
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = x.new_zeros(x.size(0))
        for block in reversed(self.blocks):
            x, logdet = block.forward(x)
            logdet_accum = logdet_accum + logdet
        return x, logdet_accum

    @overrides
    def backward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = y.new_zeros(y.size(0))
        for block in self.blocks:
            y, logdet = block.backward(y)
            logdet_accum = logdet_accum + logdet
        return y, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "AFMADE":
        return AFMADE(**params)


AFMADE.register('af_made')
