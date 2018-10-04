__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import torch
from mae.modules.flows.flow import Flow


class IAF(Flow):
    """
    Inverse Auto-Regressive Flow for 1d data. input shape (batch, nz)
    The forward path is implemented by IAF, the backward is the corresponding AF.
    """
    def __init__(self, input_size):
        super(IAF, self).__init__()
        self._input_size = input_size

    def input_size(self) -> Tuple:
        """

        Returns: a tuple of the input shape of encoded z (excluding batch_size)
            Note: the output shape of a flow should be the same as its input shape

        """
        return self._input_size,

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
    def from_params(cls, params: Dict):
        raise NotImplementedError


class IAF2d(Flow):
    """
    Inverse Auto-Regressive Flow for 2d data. input shape (batch, channels, H, W)
    The forward path is implemented by IAF, the backward is the corresponding AF.
    """
    def __init__(self, channels, H, W):
        super(IAF2d, self).__init__()
        self._input_shape = (channels, H, W)

    def input_size(self) -> Tuple:
        """

        Returns: a tuple of the input shape of encoded z (excluding batch_size)
            Note: the output shape of a flow should be the same as its input shape

        """
        return self._input_shape

    @overrides
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: Tensor
                The random variable before flow

        Returns: y: Tensor, logdet: Tensor
            y, the random variable after flow
            logdet, the log determinant of :math:`\partial x / \partial y`
            Then the density :math:`\log(p(y)) = \log(p(x)) + logdet`

        """
        raise NotImplementedError

    @overrides
    def backward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            y: Tensor
                The random variable after flow

        Returns: x: Tensor, logdet: Tensor
            x, the random variable before flow
            logdet, the log determinant of :math:`\partial x / \partial y`
            Then the density :math:`\log(p(y)) = \log(p(x)) + logdet`

        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Dict):
        raise NotImplementedError
