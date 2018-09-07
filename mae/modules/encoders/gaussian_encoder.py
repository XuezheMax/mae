__author__ = 'max'

import math
import numpy as np
from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from mae.modules.encoders.encoder import Encoder
from mae.modules.encoders.encoder_cores.encoder_core import EncoderCore
from mae.modules.flows.flow import Flow


class GaussianEncoder(Encoder):
    @classmethod
    def _CHECK(cls, core: EncoderCore, prior_flow: Flow, posterior_flow: Flow):
        def _CHECK_SIZE(s1: Tuple, s2: Tuple):
            assert np.prod(s1) == np.prod(s2)

        if prior_flow is not None:
            try:
                _CHECK_SIZE(core.output_size(), prior_flow.input_size())
            except AssertionError:
                raise ValueError('core output size %s does not match prior input size %s' % (core.output_size(), prior_flow.input_size()))

        if posterior_flow is not None:
            try:
                _CHECK_SIZE(core.output_size(), posterior_flow.input_size())
            except AssertionError:
                raise ValueError('core output size %s does not match posterior input size %s' % (core.output_size(), posterior_flow.input_size()))

    def __init__(self, core: EncoderCore,
                 prior_flow: Flow = None,
                 posterior_flow: Flow = None,
                 ngpu=1):
        super(GaussianEncoder, self).__init__()
        self.ngpu = ngpu
        GaussianEncoder._CHECK(core, prior_flow, posterior_flow)

        if ngpu > 1:
            core = nn.DataParallel(core, device_ids=list(range(ngpu)))

        self.core = core
        self.prior_flow = prior_flow
        self.posterior_flow = posterior_flow

    @overrides
    def z_shape(self) -> Tuple:
        if isinstance(self.core, EncoderCore):
            return self.core.output_size()
        else:
            assert isinstance(self.core, nn.DataParallel)
            return self.core.module.output_size()

    def reparameterize(self, mu, logvar, nsamples=1):
        # [batch, z_shape]
        z_size = mu.size()
        std = logvar.mul(0.5).exp()
        # [batch, nsamples, z_shape]
        eps = std.new_empty(z_size[0], nsamples, *z_size[1:]).normal_()
        return eps.mul(std.unsqueeze(1)).add(mu.unsqueeze(1))

    @overrides
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
        # [batch, z_shape]
        mu, logvar = self.core(x)
        # [batch, nsamples, z_shape]
        z_normal = self.reparameterize(mu, logvar, nsamples)
        z_size = z_normal.size()
        if self.posterior_flow is not None:
            # [batch * nsamples, flow_shape]
            z, logdet = self.posterior_flow.forward(z_normal.view(z_size[0] * z_size[1], *self.posterior_flow.input_size()))
            # [batch, nsamples, z_shape]
            z = z.view(*z_size)
            logdet = logdet.view(z_size[0], z_size[1])
        else:
            z = z_normal
            logdet = z.new_zeros(z_size[0], z_size[1])

        distr_parameters = (mu, logvar, z_normal, logdet)
        return z, distr_parameters

    @overrides
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

        z_size = self.z_shape()
        # [nsamples, z_shape]
        z_normal = torch.randn(nsamples, *z_size).to(device)
        if self.prior_flow is not None:
            # [nsamples, flow_shape]
            z, logdet = self.prior_flow.forward(z_normal.view(nsamples, *self.prior_flow.input_size()))
            # [nsamples, z_shape]
            z = z.view(nsamples, *z_size)
        else:
            z = z_normal
            logdet = z.new_zeros(nsamples)

        distr_paramsters = (z_normal, logdet)
        return z, distr_paramsters

    def _postKL(self, mu, logvar):
        eps = 1e-12
        z_shape = self.z_shape()
        batch_size = mu.size(0)
        # [batch, z_shape]
        var = logvar.exp()
        # A [batch, batch. z_shape]
        A = var.unsqueeze(1).div(var.unsqueeze(0) + eps)
        # B [batch, batch, z_shape]
        B = (mu.unsqueeze(1) - mu.unsqueeze(0)).pow(2).div(var.unsqueeze(0) + eps)
        # C [batch, batch, z_shape]
        C = logvar.unsqueeze(1) - logvar.unsqueeze(0)

        # [batch, batch]
        Eye = torch.eye(batch_size, device=mu.device)
        # [batch, batch, z_shape]
        PostKL = (A + B - C - 1) * 0.5
        PostKL = PostKL * (1.0 - Eye.view(Eye.size() + (1,) * len(z_shape)))

        cc = batch_size / (batch_size - 1.0)
        # [batch, batch, z_shape] --> [z_shape]
        PostKL_mean = PostKL.view(batch_size ** 2, *z_shape).mean(dim=0) * cc

        dd = math.sqrt((batch_size ** 2 - 1) / (batch_size ** 2 - batch_size - 1.0))
        # [batch, batch, z_shape] --> [batch, batch, -1] --> [batch, batch] --> [1]
        PostKL_std = (PostKL.view(Eye.size() + (-1,)).sum(dim=2) + Eye * PostKL_mean.sum()).std() * dd
        return PostKL_mean, PostKL_std

    @overrides
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
        # [batch, nsamples, z_shape]
        z, distr_params = self.sample_from_posterior(x, nsamples=nsamples)
        mu, logvar, _, _ = distr_params

        z_size = z.size()
        # [batch, nz]
        mu = mu.view(z_size[0], -1)
        logvar = logvar.view(z_size[0], -1)

        if self.prior_flow is None and self.posterior_flow is None:
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            # KL = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1)
            KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        else:
            # [batch * nsamples] --> [batch, nsamples]
            log_probs_prior = self.log_probability_prior(z.view(z_size[0] * z_size[1], *z_size[2:])).view(z_size[0], z_size[1])
            log_probs_posterior = self.log_probability_posterior(x, z, distr_params=distr_params)

            # compute KL using MC [batch]
            KL = (log_probs_posterior - log_probs_prior).mean(dim=1)

        PostKL = self._postKL(mu, logvar)

        return z, KL, PostKL

    @overrides
    def log_probability_prior(self, z, distr_params=None):
        '''

        Args:
            z: Tensor
                The tensor of z with shape [batch, z_shape]
            distr_params: Object
                The parameters of the prior distribution.

        Returns: Tensor
            The tensor of the log prior probabilities of z shape = [batch]

        '''
        if distr_params is not None:
            z_normal, logdet = distr_params
        else:
            if self.prior_flow is None:
                z_normal = z
                logdet = 0.
            else:
                z_normal, logdet = self.prior_flow.backward(z)
        # [batch, z_shape]
        log_probs = z_normal.pow(2) + math.log(math.pi * 2.)
        # [batch, z_shape] --> [batch, nz] -- > [batch]
        return log_probs.view(z.size(0), -1).sum(dim=1) * -0.5 + logdet

    @overrides
    def log_probability_posterior(self, x, z, distr_params=None):
        '''

        Args:
            x: Tensor
                The input data with shape =[batch, *]
            z: Tensor
                The tensor of z with shape [batch, nsamples, z_shape]
            distr_params: Object
                The parameters of the posterior distribution.

        Returns: Tensor
            The tensor of the log posterior probabilities of z|x shape = [batch, nsamples]

        '''
        eps = 1e-12
        z_size = z.size()
        if distr_params is None:
            mu, logvar = self.core(x)
            if self.posterior_flow is None:
                z_normal = z
                logdet = z.new_zeros(z_size[0], z_size[1])
            else:
                z_normal, logdet = self.posterior_flow.backward(z)
        else:
            mu, logvar, z_normal, logdet = distr_params

        # [batch, nsamples, z_shape]
        log_probs = logvar.unsqueeze(1) + (z_normal - mu.unsqueeze(1)).pow(2).div(logvar.exp().unsqueeze(1) + eps) + math.log(math.pi * 2.)
        # [batch, nsamples, nz] --> [batch, nsamples]
        log_probs = log_probs.view(z_size[0], z_size[1], -1).sum(dim=2) * -0.5 + logdet
        return log_probs

    @classmethod
    def from_params(cls, params: Dict) -> "GaussianEncoder":
        core_params = params.pop('core')
        core = EncoderCore.by_name(core_params.pop('type')).from_params(core_params)

        posterior_flow = None
        posterior_flow_params = params.pop('posterior_flow', None)
        if posterior_flow_params is not None:
            posterior_flow = Flow.by_name(posterior_flow_params.pop('type')).from_params(posterior_flow_params)

        prior_flow = None
        prior_flow_params = params.pop('prior_flow', None)
        if prior_flow_params is not None:
            prior_flow = Flow.by_name(prior_flow_params.pop('type')).from_params(prior_flow_params)

        return GaussianEncoder(core,
                               prior_flow=prior_flow,
                               posterior_flow=posterior_flow,
                               **params)


GaussianEncoder.register('gaussian')
