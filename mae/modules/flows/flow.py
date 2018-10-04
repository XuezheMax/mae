__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn


class Flow(nn.Module):
    """
    Normalizing Flow base class
    """
    _registry = dict()

    def __init__(self):
        super(Flow, self).__init__()

    def input_size(self) -> Tuple:
        """

        Returns: a tuple of the input shape of encoded z (excluding batch_size)
            Note: the output shape of a flow should be the same as its input shape

        """
        raise NotImplementedError

    @overrides
    def forward(self, x: torch.Tensor, init=False, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: Tensor
                The random variable before flow
            init: bool
                perform initialization or not (default: False)
            init_scale: float
                initial scale (default: 1.0)

        Returns: y: Tensor, logdet: Tensor
            y, the random variable after flow
            logdet, the log determinant of :math:`\partial x / \partial y`
            Then the density :math:`\log(p(y)) = \log(p(x)) + logdet`

        """
        raise NotImplementedError

    def backward(self, y: torch.Tensor, init=False, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            y: Tensor
                The random variable after flow
            init: bool
                perform initialization or not (default: False)
            init_scale: float
                initial scale (default: 1.0)

        Returns: x: Tensor, logdet: Tensor
            x, the random variable before flow
            logdet, the log determinant of :math:`\partial x / \partial y`
            Then the density :math:`\log(p(y)) = \log(p(x)) + logdet`

        """
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        Flow._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return Flow._registry[name]

    @classmethod
    def from_params(cls, params: Dict):
        raise NotImplementedError
