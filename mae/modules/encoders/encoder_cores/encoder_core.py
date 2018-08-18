__author__ = 'max'

from typing import Dict, Tuple
import torch.nn as nn


class EncoderCore(nn.Module):
    """
    Encoder Core base class
    """
    _registry = dict()

    def __init__(self):
        super(EncoderCore, self).__init__()

    def output_size(self) -> Tuple:
        """

        Returns: a tuple of the output shape of encoded z (excluding batch_size)

        """
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        EncoderCore._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return EncoderCore._registry[name]

    @classmethod
    def from_params(cls, params: Dict):
        raise NotImplementedError
