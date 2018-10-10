__author__ = 'max'

from typing import Dict, Tuple
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder base class
    """
    _registry = dict()

    def __init__(self):
        super(Encoder, self).__init__()

    def z_shape(self) -> Tuple:
        raise NotImplementedError

    def initialize(self, x, init_scale=1.0):
        """

        Args:
            x: Tensor
                The input data used for initialization
            init_scale: float
                initial scale
        Returns: Tensor
            the tensor of output

        """
        raise NotImplementedError

    def sample_from_posterior(self, x, nsamples=1):
        """

        Args:
            x: Tensor
                The input data with shape =[batch, *]
            nsample: int
                Number of samples for each data instance

        Returns: Tensor, Object
            Tensor: the tensor of samples from the posterior distribution with shape [batch, nsamples, z_shape]
            Object: parameters associated with the posterior distribution
        """
        raise NotImplementedError

    def sample_from_proir(self, nsamples=1, device=torch.device('cpu')):
        """

        Args:
            nsamples: int
                Number of samples for each data instance
            device: torch.device
                the device to store the samples

        Returns: Tensor
            the tensor of samples from the posterior distribution with shape [nsamples, z_shape]

        """
        raise NotImplementedError

    def encode(self, x, nsamples):
        """

        Args:
            x: Tensor
                The input data with shape =[batch, *]
            nsample: int
                Number of samples for each data instance

        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: the tensor with latent z for x shape [batch, nsamples, z_shape]
            Tensor2: the tensor of KL for each x [batch]
            (Tensor3, Tensor4): the tensors of posterior measures for each pair of x with shape [z_shape], [1]

        """
        raise NotImplementedError

    def log_probability_prior(self, z):
        """

        Args:
            z: Tensor
                The tensor of z with shape [batch, z_shape]

        Returns: Tensor
            The tensor of the log prior probabilities of z shape = [batch]

        """
        raise NotImplementedError

    def log_probability_posterior(self, x, z, distr_params=None):
        """

        Args:
            x: Tensor
                The input data with shape =[batch, *]
            z: Tensor
                The tensor of z with shape [batch, nsamples, z_shape]
            distr_params: Object
                The parameters of the posterior distribution.

        Returns: Tensor
            The tensor of the log posterior probabilities of z|x shape = [batch, nsamples]

        """
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        Encoder._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return Encoder._registry[name]

    @classmethod
    def from_params(cls, params: Dict):
        raise NotImplementedError
