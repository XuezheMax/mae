__author__ = 'max'

import torch.nn as nn
from typing import Dict

from mae.modules.networks.auto_regressives.made import MADE
from mae.modules.flows.iaf.iaf import IAF


class IAFMADEBlock(nn.Module):
    def __init__(self, nz, num_hiddens, hidden_size, order, bias=True):
        super(IAFMADEBlock, self).__init__()
        self.mu = MADE(nz, num_hiddens, hidden_size, order, bias=bias)
        self.logvar = MADE(nz, num_hiddens, hidden_size, order, bias=bias)

    def forward(self, z):
        # [batch, nz]
        mu = self.mu(z)
        logstd = self.logvar(z) * 0.5
        z = mu + z * logstd.exp()
        return z, logstd.sum(dim=1)


class IAFMADEDualEBlock(nn.Module):
    def __init__(self, nz, num_hiddens, hidden_size, bias=True):
        super(IAFMADEDualEBlock, self).__init__()
        self.fwd = IAFMADEBlock(nz, num_hiddens, hidden_size, 'A', bias=bias)
        self.bwd = IAFMADEBlock(nz, num_hiddens, hidden_size, 'B', bias=bias)

    def forward(self, z):
        # forward
        # [batch, nz]
        z, logdet_fwd = self.fwd(z)
        # backward
        z, logdet_bwd = self.bwd(z)
        return z, logdet_fwd + logdet_bwd


class IAFMADE(IAF):
    def __init__(self, nz, num_blocks, num_hiddens, hidden_size, bias=True):
        super(IAFMADE, self).__init__(nz)
        self.num_blocks = num_blocks
        self.num_hiddens = num_hiddens
        self.hidden_size = hidden_size
        self.blocks = []
        for i in range(num_blocks):
            block = IAFMADEDualEBlock(self.nz, num_hiddens, hidden_size, bias=bias)
            self.blocks.append(block)
        assert num_blocks == len(self.blocks)
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, z):
        logdet_accum = z.new_zeros(z.size(0))
        for block in self.blocks:
            z, logdet = block(z)
            logdet_accum = logdet_accum + logdet
        return z, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "IAFMADE":
        return IAFMADE(**params)


IAFMADE.register('iaf_made')
