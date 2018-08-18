__author__ = 'max'

import torch
import torch.nn as nn


class EncoderCore(nn.Module):
    """
    Encoder Core base class
    """
    def __init__(self):
        super(EncoderCore, self).__init__()

    def output_size(self):
        """

        Returns: a tuple of the output shape of encoded z (excluding batch_size)

        """
        raise NotImplementedError
