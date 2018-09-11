__author__ = 'max'

from typing import Dict, Tuple
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F

from mae.modules.decoders.decoder import Decoder
from mae.modules.utils import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic


class ColorImageDecoder(Decoder):
    """
    Base class for color image decoder.
    The probabilistic distribution is mixure of discretized logistic distribution.
    """

    def __init__(self, nmix, ngpu=1):
        super(ColorImageDecoder, self).__init__()
        self.nc = 3
        self.nmix = nmix
        self.ngpu = ngpu

    @overrides
    def z_shape(self) -> Tuple:
        raise NotImplementedError

    @overrides
    def output_size(self) -> Tuple:
        """

        Returns: a tuple of the output shape of decoded x (excluding batch_size)

        """
        raise NotImplementedError

    def execute(self, z, x):
        # [batch, (2 * nc + 4) * nmix, H, W]
        output = self(x, z)

        # [batch, mix, H, W]
        logit_probs = output[:, :self.nmix]

        batch, nmix, H, W = logit_probs.size()

        # [batch, nc * mix, H, W] --> [batch, mix, nc, H, W]
        mu = torch.tanh(output[:, nmix:(self.nc + 1) * nmix].view(batch, nmix, self.nc, H, W))
        log_scale = output[:, (self.nc + 1) * nmix:(self.nc * 2 + 1) * nmix].view(batch, nmix, self.nc, H, W).clamp(min=-7.0)
        coeffs = torch.tanh(output[:, (self.nc * 2 + 1) * nmix:(self.nc * 2 + 4) * nmix].view(batch, nmix, self.nc, H, W))

        return mu, log_scale, F.log_softmax(logit_probs, dim=1), coeffs

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
            the probability matrix of each pixel shape=[batch, x_shape] Note: for RGB image we do not compute the probability (None)

        """
        x = None
        # [batch, mix, nc, H, W]
        mu, log_scale, logit_probs, coeffs = self.execute(z, x)
        return sample_from_discretized_mix_logistic(mu, log_scale, coeffs, logit_probs, random_sample), None

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
        # [batch, mix, nc, H, W]
        z_size = z.size()
        batch_size, nsampels = z_size[:2]
        # [batch * nsamples, mix, x_shape]
        mu, log_scale, logit_probs, coeffs = self.execute(z.view(batch_size * nsampels, *z_size[2:]), x)
        # [batch, nsamples, mix, x_shape]
        mu = mu.view(batch_size, nsampels, *mu.size()[1:])
        log_scale = log_scale.view(batch_size, nsampels, *log_scale.size()[1:])
        coeffs = coeffs.view(batch_size, nsampels, *coeffs.size()[1:])
        logit_probs = logit_probs.view(batch_size, nsampels, *logit_probs.size()[1:])

        bin_size = 1. / 255.
        lower = 1. / 255. - 1.0
        upper = 1.0 - 1. / 255.
        # [batch, nc, H, W] --> [batch, 1, nc, H, W] --> [batch, 1, 1, nc, H, W]
        x_reshape = x.unsqueeze(1).unsqueeze(1)
        return discretized_mix_logistic_loss(x_reshape, mu, log_scale, coeffs, bin_size, lower, upper, logit_probs)

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
    def from_params(cls, params: Dict) -> "ColorImageDecoder":
        raise NotImplementedError