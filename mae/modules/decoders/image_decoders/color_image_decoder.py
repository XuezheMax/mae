__author__ = 'max'

from typing import Dict, Tuple
from overrides import overrides
import torch
import torch.nn as nn

from mae.modules.decoders.decoder import Decoder


class ColorImageDecoder(Decoder):
    """
    Base class for color image decoder.
    The probabilistic distribution is mixure of discretized logistic distribution.
    """

    def __init__(self, nz, ngpu=1):
        super(ColorImageDecoder, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.core = nn.Module()