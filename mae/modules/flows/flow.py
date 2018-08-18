__author__ = 'max'

import torch.nn as nn


class Flow(nn.Module):
    def __init__(self, nz):
        super(Flow, self).__init__()
        self.nz = nz

    def input_size(self):
        """

        Returns: a tuple of the input shape of encoded z (excluding batch_size)
            Note: the output shape of a flow should be the same as its input shape

        """
        raise NotImplementedError
