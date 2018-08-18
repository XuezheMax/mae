__author__ = 'max'

from typing import Dict, Tuple
import torch.nn as nn


class Flow(nn.Module):
    """
    Normalizing Flow base class
    """
    _registry = dict()

    def __init__(self, nz):
        super(Flow, self).__init__()
        self.nz = nz

    def input_size(self) -> Tuple:
        """

        Returns: a tuple of the input shape of encoded z (excluding batch_size)
            Note: the output shape of a flow should be the same as its input shape

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
