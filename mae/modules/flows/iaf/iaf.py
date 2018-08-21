__author__ = 'max'

import torch.nn as nn
from typing import Dict, Tuple
from mae.modules.flows.flow import Flow


class IAF(Flow):
    """
    Inverse Auto-Regressive Flow for 1d data. input shape (batch, nz)
    """
    def __init__(self, nz):
        super(IAF, self).__init__()
        self.nz = nz

    def input_size(self) -> Tuple:
        """

        Returns: a tuple of the input shape of encoded z (excluding batch_size)
            Note: the output shape of a flow should be the same as its input shape

        """
        return self.nz,


class IAF2d(Flow):
    """
    Inverse Auto-Regressive Flow for 2d data. input shape (batch, channels, H, W)
    """
    def __init__(self, channels, H, W):
        super(IAF2d, self).__init__()
        self.z_shape = (channels, H, W)

    def input_size(self) -> Tuple:
        """

        Returns: a tuple of the input shape of encoded z (excluding batch_size)
            Note: the output shape of a flow should be the same as its input shape

        """
        return self.z_shape
