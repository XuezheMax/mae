__author__ = 'max'

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder base class
    """
    def __init__(self):
        super(Encoder, self).__init__()

    def sample_from_posterior(self, x, nsamples=1):
        '''

        Args:
            x: Tensor
                The input data with shape =[batch, *]
            nsample: int
                Number of samples for each data instance

        Returns: Tensor, Object
            Tensor: the tensor of samples from the posterior distribution with shape [batch, nsamples, nz]
            Object: parameters associated with the posterior distribution
        '''
        raise NotImplementedError

    def sample_from_proir(self, nsamples=1, device=torch.device('cpu')):
        '''

        Args:
            nsamples: int
                Number of samples for each data instance
            device: torch.device
                the device to store the samples

        Returns: Tensor
            the tensor of samples from the posterior distribution with shape [nsamples, nz]

        '''
        raise NotImplementedError

    def encode(self, x, nsamples):
        '''

        Args:
            x: Tensor
                The input data with shape =[batch, *]
            nsample: int
                Number of samples for each data instance

        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: the tensor with latent z for x shape [batch, nsamples, nz]
            Tensor2: the tensor of KL for each x [batch]
            Tensor3: the tensor of HLG measures for each pair of x with shape [batch, batch]

        '''
        raise NotImplementedError

    def log_probability_prior(self, z):
        '''

        Args:
            z: Tensor
                The tensor of z with shape [batch, nz]

        Returns: Tensor
            The tensor of the log prior probabilities of z shape = [batch]

        '''
        raise NotImplementedError

    def log_probability_posterior(self, x, z, distr_params=None):
        '''

        Args:
            x: Tensor
                The input data with shape =[batch, *]
            z: Tensor
                The tensor of z with shape [batch, nsamples, nz]
            distr_params: Object
                The parameters of the posterior distribution.

        Returns: Tensor
            The tensor of the log posterior probabilities of z|x shape = [batch, nsamples]

        '''
        raise NotImplementedError
