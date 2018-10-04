__author__ = 'max'

import os
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from mae.modules.encoders import Encoder
from mae.modules.decoders import Decoder
from mae.modules.utils import logsumexp


class MAE(nn.Module):
    @classmethod
    def _CHECK(cls, encoder: Encoder, decoder: Decoder):
        def _CHECK_SIZE(s1: Tuple, s2: Tuple):
            assert np.prod(s1) == np.prod(s2)

        try:
            _CHECK_SIZE(encoder.z_shape(), decoder.z_shape())
        except AssertionError:
            raise ValueError('encoder latent z shape %s does not match decoder input shape %s' % (encoder.z_shape(), decoder.z_shape()))

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(MAE, self).__init__()
        MAE._CHECK(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def z_shape(self) -> Tuple:
        return self.encoder.z_shape()

    def sample_from_proir(self, nsamples=1, device=torch.device('cpu')):
        """

        Args:
            nsamples: int
                Number of samples for each data instance
            device: torch.device
                The device to store the samples

        Returns: Tensor, Object
            Tensor: the tensor of samples from the posterior distribution with shape [nsamples, z_shape]
            Object: parameters associated with the posterior distribution
        """
        return self.encoder.sample_from_proir(nsamples, device=device)

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
        return self.encoder.sample_from_posterior(x, nsamples)

    def encode(self, x, nsamples):
        """

        Args:
            x: Tensor
                The input data with shape =[batch, *]
            nsample: int
                Number of samples for each data instance

        Returns: Tensor1, Tensor2, (Tensor3, Tensor4)
            Tensor1: the tensor with latent z for x shape [batch, nsamples, z_shape]
            Tensor2: the tensor of KL for each x [batch]
            (Tensor3, Tensor4): the tensors of posterior measures for each pair of x with shape [z_shape], [1]

        """
        return self.encoder.encode(x, nsamples)

    def decode(self, z, random_sample):
        """
        Args:
            z: Tensor
                the tensor of latent z shape=[batch, z_shape]
            random_sample: boolean
                randomly sample or decode via argmaximizing probability

        Returns: Tensor
            the tensor of decoded x shape=[batch, x_shape]
            the probability matrix of each pixel shape=[batch, x_shape]

        """
        z_shape_dec = self.decoder.z_shape()
        return self.decoder.decode(z.view(z.size(0), *z_shape_dec), random_sample)

    def loss(self, x, nsamples, eta=0.0, gamma=0.0, free_bits=0.0):
        """

        Args:
            x: Tensor
                The input data with shape =[batch, *]
            nsample: int
                Number of samples for each data instance
            eta: float
                weight of posterior KL mean
            gamma: float
                weight of posterior KL std
            free_bits: float
                free bits for KL

        Returns: Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6, Tensor7
            Tensor1: loss objective shape=[1]
            Tensor2: reconstruction loss shape=[batch]
            Tensor3: the tensor of KL for each x [batch]
            Tensor4: mean of posterior KL shape=[1]
            Tensor5: std of posterior KL shape=[1]
            Tensor6: loss of mean of posterior KL shape=[1]
            Tensor7: loss of std of posterior KL shape=[1]

        """
        # [batch, nsamples, z_shape]
        z, KL, postKL = self.encode(x, nsamples)

        z_shape_dec = self.decoder.z_shape()
        # [batch, nsamples]
        reconstruct_err = self.decoder.reconstruct_error(x, z.view(z.size(0), z.size(1), *z_shape_dec))

        postKL_mean = postKL[0].sum()
        postKL_std = postKL[1]
        loss_postKL_mean = - eta * F.logsigmoid(postKL[0] / eta).sum() if eta > 0. else 0.
        loss_postKL_std = gamma * postKL_std

        recon = reconstruct_err.mean(dim=1)

        loss = recon.mean() + KL.mean().clamp(min=free_bits) + loss_postKL_mean + loss_postKL_std
        return loss, recon, KL, postKL_mean, postKL_std, loss_postKL_mean, loss_postKL_std

    def reconstruct(self, x, k=50, random_sample=False):
        """

        Args:
            x: Tensor
                The input data with shape =[batch, *]
            k: int
                the number of samples for reconstruction
            random_sample: boolean
                randomly sample or decode via argmaximizing probability

        Returns: Tensor
            the tensor of reconstruct data [batch, *]
            the probability matrix of each pixel shape=[batch, x_shape]

        """
        # [batch, k, z_shape]
        z, _ = self.sample_from_posterior(x, nsamples=k)
        # [batch, k, z_shape] -> [batch, z_shape] --> [batch, x_shape]
        return self.decode(z.mean(dim=1), random_sample=random_sample)

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
        z = self.encoder.initialize(x, init_scale=init_scale)
        z_shape_dec = self.decoder.z_shape()
        return self.decoder.initialize(x, z.view(z.size(0), *z_shape_dec), init_scale=init_scale)

    def nll(self, x, k):
        """
        compute negative log-likelihood via importance weighted elbo.

        Args:
            x: Tensor
                The input data with shape =[batch, *]
            k: int
                The number of samples for importance weighted MC

        Returns: (Tensor11, Tensor12, Tensor13), Tensor2
            Tensor11: nll with ELBO estimation [batch]
            Tensor12: reconstruction error [batch]
            Tensor13: kl [batch]
            Tensor2: nll with Importance Weighted estimation [batch]

        """
        batch = x.size(0)
        # [batch, k, z_shape]
        z, distr_params = self.sample_from_posterior(x, nsamples=k)
        z_size = self.z_shape()
        z_shape_dec = self.decoder.z_shape()

        # [batch, k, z_shape] --> [batch * k, z_shape] --> [batch * k] --> [batch, k]
        log_probs_prior = self.encoder.log_probability_prior(z.view(batch * k, *z_size)).view(batch, k)
        # [batch, k]
        log_probs_posterior = self.encoder.log_probability_posterior(x, z, distr_params=distr_params)
        log_probs_gen = self.decoder.log_probability(x, z.view(batch, k, *z_shape_dec))

        # [batch, k]
        log_iw = log_probs_gen + log_probs_prior - log_probs_posterior

        # [batch, k] --> [batch]
        nll_elbo = log_iw.mean(dim=1) * -1.
        recon_err = log_probs_gen.mean(dim=1) * -1.
        kl = (log_probs_posterior - log_probs_prior).mean(dim=1)
        nll_iw = logsumexp(log_iw, dim=1) * -1. + math.log(k)
        return (nll_elbo, recon_err, kl), nll_iw

    @classmethod
    def from_params(cls, params: Dict) -> "MAE":
        encoder_params = params.pop('encoder')
        encoder = Encoder.by_name(encoder_params.pop('type')).from_params(encoder_params)

        decoder_params = params.pop('decoder')
        decoder = Decoder.by_name(decoder_params.pop('type')).from_params(decoder_params)

        return MAE(encoder, decoder)

    @classmethod
    def load(cls, model_path, device) -> "MAE":
        params = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
        model_name = os.path.join(model_path, 'model.pt')
        mae = MAE.from_params(params)
        mae.load_state_dict(torch.load(model_name, map_location=device))
        return mae.to(device)
