__author__ = 'max'

from overrides import overrides
import torch.nn as nn
import torch
from typing import Dict, Tuple, List

from mae.modules.networks.auto_regressives.made import MADE
from mae.modules.flows.iaf.iaf import IAF


class IAFMADEBlock(nn.Module):
    def __init__(self, input_size, num_hiddens, hidden_size, order, bias=True, var=True):
        super(IAFMADEBlock, self).__init__()
        self.mu = MADE(input_size, num_hiddens, hidden_size, order, bias=bias)
        if var:
            self.logvar = MADE(input_size, num_hiddens, hidden_size, order, bias=bias)
        else:
            self.logvar = None
        assert input_size > 0, 'input size (%s) should be positive' % input_size
        self._input_size = input_size

    def initialize(self, x, init_scale=1.0):
        # [batch, nz]
        mu = self.mu.initialize(x, init_scale=init_scale)
        if self.logvar:
            logstd = self.logvar.initialize(x, init_scale=init_scale) * 0.5
        else:
            logstd = mu.new_zeros(mu.size())
        y = mu + x * logstd.exp()
        return y, logstd.sum(dim=1) * -1.0

    def forward(self, x):
        # [batch, nz]
        mu = self.mu(x)
        if self.logvar:
            logstd = self.logvar(x) * 0.5
        else:
            logstd = mu.new_zeros(mu.size())
        y = mu + x * logstd.exp()
        return y, logstd.sum(dim=1) * -1.0

    def backward(self, y):
        eps = 1e-12
        x = y.new_zeros(y.size())
        logstd = y.new_zeros(y.size())
        for _ in range(self._input_size):
            mu = self.mu(x)
            if self.logvar:
                logstd = self.logvar(x) * 0.5
            x = (y - mu).div(logstd.exp() + eps)
        return x, logstd.sum(dim=1) * -1.0


class IAFMADEDualBlock(nn.Module):
    def __init__(self, input_size, num_hiddens, hidden_size, bias=True, var=True):
        super(IAFMADEDualBlock, self).__init__()
        self.fwd = IAFMADEBlock(input_size, num_hiddens, hidden_size, 'A', bias=bias, var=var)
        self.bwd = IAFMADEBlock(input_size, num_hiddens, hidden_size, 'B', bias=bias, var=var)

    def initialize(self, x, init_scale=1.0):
        # forward
        # [batch, nz]
        x, logdet_fwd = self.fwd.initialize(x, init_scale=init_scale)
        # backward
        y, logdet_bwd = self.bwd.initialize(x, init_scale=init_scale)
        return y, logdet_fwd + logdet_bwd

    def forward(self, x):
        # forward
        # [batch, nz]
        x, logdet_fwd = self.fwd.forward(x)
        # backward
        y, logdet_bwd = self.bwd.forward(x)
        return y, logdet_fwd + logdet_bwd

    def backward(self, y):
        # backward
        y, logdet_bwd = self.bwd.backward(y)
        x, logdet_fwd = self.fwd.backward(y)
        return x, logdet_fwd + logdet_bwd


class IAFMADE(IAF):
    def __init__(self, input_size, num_blocks, num_hiddens=1, hidden_size=None, bias=True, var=True):
        super(IAFMADE, self).__init__(input_size)
        self.num_blocks = num_blocks
        self.num_hiddens = num_hiddens
        hidden_size = input_size * 10 if hidden_size is None else hidden_size
        self.hidden_size = hidden_size
        self.blocks = []
        for i in range(num_blocks):
            block = IAFMADEDualBlock(self.nz, num_hiddens, hidden_size, bias=bias, var=var)
            self.blocks.append(block)
        assert num_blocks == len(self.blocks)
        self.blocks = nn.ModuleList(self.blocks)

    @overrides
    def forward(self, x: torch.Tensor, init=False, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = x.new_zeros(x.size(0))
        for block in self.blocks:
            x, logdet = block.initialize(x, init_scale=init_scale) if init else block.forward(x)
            logdet_accum = logdet_accum + logdet
        return x, logdet_accum

    @overrides
    def backward(self, y: torch.Tensor, init=False, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        assert not init, 'IAF does not support initialization via backward.'
        logdet_accum = y.new_zeros(y.size(0))
        for block in reversed(self.blocks):
            y, logdet = block.backward(y)
            logdet_accum = logdet_accum + logdet
        return y, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "IAFMADE":
        return IAFMADE(**params)


IAFMADE.register('iaf_made')
