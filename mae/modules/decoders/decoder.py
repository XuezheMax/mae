__author__ = 'max'

from typing import Dict, Tuple
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder base class
    """
    _registry = dict()

    def __init__(self):
        super(Decoder, self).__init__()

    def z_shape(self) -> Tuple:
        """

        Returns: the shape of input z

        """
        raise NotImplementedError

    def output_size(self) -> Tuple:
        """

        Returns: a tuple of the output shape of decoded x (excluding batch_size)

        """
        raise NotImplementedError

    def initialize(self, x, z, init_scale=1.0):
        """

        Args:
            x: Tensor
                The input data used for initialization, shape [batch, x_shape]
            z: Tensor
                The input latent code used for initialization, shape = [batch, z_shape]
            init_scale: float
                initial scale
        Returns: Tensor
            the tensor of output

        """
        raise NotImplementedError

    def decode(self, z, random_sample):
        """
        Args:
            z: Tensor
                the tensor of latent z shape=[batch, z_shape]
            random_sample: boolean
                randomly sample or decode via argmaximizing probability
                the probability matrix of each pixel shape=[batch, x_shape]

        Returns: Tensor
            the tensor of decoded x shape=[batch, x_shape]

        """
        raise NotImplementedError

    def reconstruct_error(self, x, z):
        """
        Args:
            x: Tensor
                The input data with shape = [batch, x_shape]
            z: Tensor
                the tensor of latent z shape=[batch, nsamples, z_shape]

        Returns: Tensor
            the tensor of reconstruction error of x shape=[batch, nsamples]

        """
        raise NotImplementedError

    def log_probability(self, x, z):
        """
        Args:
            x: Tensor
                The input data with shape =[batch, x_shape]
            z: Tensor
                The tensor of z with shape [batch, nsamples, z_shape]

        Returns: Tensor
            The tensor of the posterior probabilities of z|x shape = [batch, nsamples]

        """
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        Decoder._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return Decoder._registry[name]

    @classmethod
    def from_params(cls, params: Dict) -> "Decoder":
        raise NotImplementedError
