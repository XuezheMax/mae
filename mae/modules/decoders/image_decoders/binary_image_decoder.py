__author__ = 'max'

from typing import Dict, Tuple
from overrides import overrides
import torch
import torch.nn as nn

from mae.modules.decoders.decoder import Decoder


class BinaryImageDecoder(Decoder):
    """
    Base class for binary image decoder.
    The probabilistic distribution is pixel-wise independent bernoulli.
    """

    def __init__(self, nz, ngpu=1):
        super(BinaryImageDecoder, self).__init__()
        self.nz = nz
        self.ngpu = ngpu

    @overrides
    def z_shape(self) -> Tuple:
        return self.nz, 1, 1

    @overrides
    def decode(self, z, random_sample):
        """
        This decoding method only fits to independent distributions, not for auto-regressive decoders
        Args:
            z: Tensor
                the tensor of latent z shape=[batch, z_shape]
            random_sample: boolean
                randomly sample or decode via argmaximizing probability

        Returns: Tensor, Tensor
            the tensor of decoded x shape=[batch, x_shape]
            the probability matrix of each pixel shape=[batch, x_shape]

        """
        img_probs = self(z)
        img = torch.bernoulli(img_probs) if random_sample else torch.ge(img_probs, 0.5).float()
        return img, img_probs

    @overrides
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
        eps = 1e-12
        z_size = z.size()
        batch_size, nsampels = z_size[:2]
        # [batch * nsamples, x_shape] --> [batch, nsamples, -1]
        recon_x = self(z.view(batch_size * nsampels, *z_size[2:])).view(batch_size, nsampels, -1)
        # [batch, -1]
        x_flat = x.view(batch_size, -1)
        BCE = (recon_x + eps).log() * x_flat.unsqueeze(1) + (1.0 - recon_x + eps).log() * (1. - x_flat).unsqueeze(1)
        # [batch, nsamples]
        return BCE.sum(dim=2) * -1.0

    @overrides
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
        return self.reconstruct_error(x, z) * -1.

    @classmethod
    def from_params(cls, params: Dict) -> "BinaryImageDecoder":
        raise NotImplementedError
