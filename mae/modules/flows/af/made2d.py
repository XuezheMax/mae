__author__ = 'max'

from overrides import overrides
import torch.nn as nn
import torch
from typing import Dict, Tuple

from mae.modules.flows.af.af import AF2d
from mae.modules.networks.auto_regressives.made import MADE2d


class AF2dMADEBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, num_hiddens, hidden_channels, hidden_kernels, order, bias=True, var=True):
        super(AF2dMADEBlock, self).__init__()
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
        for _ in range(H):
            for _ in range(W):
                mu = self.mu(y)
                if self.logvar:
                    logstd = self.logvar(y) * 0.5
                y = (x - mu).div(logstd.exp() + eps)
        return y, logstd.view(x.size(0), -1).sum(dim=1)

    def backward(self, y):
        # [batch, x_shape]
        mu = self.mu(y)
        if self.logvar:
            logstd = self.logvar(y) * 0.5
        else:
            logstd = mu.new_zeros(mu.size())
        x = mu + y * logstd.exp()
        print('y')
        print(y.max())
        print(y.min())
        print('--------------------------')
        if y.max().item() != y.max().item():
            import sys
            sys.exit(0)

        # print('mu')
        # print(mu.max())
        # print(mu.min())
        # print('logstd')
        # print(logstd.max())
        # print(logstd.min())
        # print('x')
        # print(x.max())
        # print(x.min())
        return x, logstd.view(mu.size(0), -1).sum(dim=1)


class AF2dMADEDualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, num_hiddens, hidden_channels, hidden_kernels, bias=True, var=True):
        super(AF2dMADEDualBlock, self).__init__()
        self.fwd = AF2dMADEBlock(in_channels, kernel_size, num_hiddens, hidden_channels, hidden_kernels, 'A', bias=bias, var=var)
        self.bwd = AF2dMADEBlock(in_channels, kernel_size, num_hiddens, hidden_channels, hidden_kernels, 'B', bias=bias, var=var)

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


class AF2dMADE(AF2d):
    """
    Auto-regressive flow with 2d input implemented with MADE2d (via masked conv2d)
    """

    def __init__(self, in_channels, height, width, num_blocks, kernel_size, num_hiddens=1, hidden_channels=None, hidden_kernels=None, bias=True, var=True):
        super(AF2dMADE, self).__init__(in_channels, height, width)
        hidden_channels = in_channels * 8 if hidden_channels is None else hidden_channels
        hidden_kernels = [kernel_size] * num_hiddens if hidden_kernels is None else hidden_kernels
        self.blocks = []
        for i in range(num_blocks):
            block = AF2dMADEDualBlock(in_channels, kernel_size, num_hiddens, hidden_channels, hidden_kernels, bias=bias, var=var)
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
    def from_params(cls, params: Dict) -> "AF2dMADE":
        return AF2dMADE(**params)


AF2dMADE.register('af2d_made')
