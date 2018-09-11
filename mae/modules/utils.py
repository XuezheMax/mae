__author__ = 'max'

import math
import numpy as np
import torch
import torch.nn.functional as F


def norm(p: torch.Tensor, dim: int):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return norm(p.transpose(0, dim), 0).transpose(0, dim)


def logsumexp(x, dim=None):
    """
    Args:
        x: A pytorch tensor (any dimension will do)
        dim: int or None, over which to perform the summation. `None`, the
             default, performs over all axes.
    Returns: The result of the log(sum(exp(...))) operation.
    """
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        return xmax_ + torch.log(torch.exp(x - xmax).sum())
    else:
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))


def logavgexp(x, dim=None):
    """
    Args:
        x: A pytorch tensor (any dimension will do)
        dim: int or None, over which to perform the summation. `None`, the
             default, performs over all axes.
    Returns: The result of the log(sum(exp(...))) operation.
    """
    if dim is None:
        xmax = x.max()
        xmax_ = x.max()
        nsize = np.prod(*x.size())
        return xmax_ + torch.log(torch.exp(x - xmax).sum()) - np.log(nsize)
    else:
        xmax, _ = x.max(dim, keepdim=True)
        xmax_, _ = x.max(dim)
        nsize = x.size(dim)
        return xmax_ + torch.log(torch.exp(x - xmax).sum(dim)) - np.log(nsize)


def discretized_mix_logistic_loss(x, means, logscales, bin_size, lower, upper, logit_probs):
    """

    Args:
        x: [batch, 1, 1, nc, H, W]
        means: [batch, nsamples, nmix, nc, H, W]
        logscales: [batch, nsamples, nmix, nc, H, W]
        bin_size: float
        lower: float
        upper: float
        logit_probs:, [batch, nsamples, nmix, H, W]

    Returns:
        loss [batch, nsamples]

    """
    eps = 1e-12
    # [batch, nsamples, nmix, nc, H, W]
    centered_x = x - means
    if isinstance(logscales, float):
        inv_stdv = np.exp(-logscales)
    else:
        inv_stdv = torch.exp(-logscales)

    # [batch, nsamples, nmix, nc, H, W]
    min_in = inv_stdv * (centered_x - bin_size)
    plus_in = inv_stdv * (centered_x + bin_size)
    x_in = inv_stdv * centered_x

    # [batch, nsamples, nmix, nc, H, W]
    cdf_min = torch.sigmoid(min_in)
    cdf_plus = torch.sigmoid(plus_in)
    # lower < x < upper
    cdf_delta = cdf_plus - cdf_min
    log_cdf_mid = torch.log(cdf_delta + eps)
    log_cdf_approx = x_in - logscales - 2. * F.softplus(x_in) + np.log(2 * bin_size)

    # x < lower
    log_cdf_low = plus_in - F.softplus(plus_in)

    # x > upper
    log_cdf_up = -F.softplus(min_in)

    # [batch, nsamples, nmix, nc, H, W]
    mask_delta = cdf_delta.gt(1e-5).float()
    log_cdf = log_cdf_mid * mask_delta + log_cdf_approx * (1.0 - mask_delta)
    mask_lower = x.ge(lower).float()
    mask_upper = x.le(upper).float()
    log_cdf = log_cdf_low * (1.0 - mask_lower) + log_cdf * mask_lower
    log_cdf = log_cdf_up * (1.0 - mask_upper) + log_cdf * mask_upper

    # [batch, nsample, nmix, nc, H, W] -> [batch, nsamples, nmix, H, W] -> [batch, nsamples, H, W]
    loss = logsumexp(log_cdf.sum(dim=3) + logit_probs, dim=2)
    # [batch, nsamples, H, W] -> [batch, nsamples, -1] -> [batch, nsamples]
    return loss.view(*loss.size()[:2], -1).sum(dim=2) * -1.



def sample_from_discretized_mix_logistic(means, logscales, logit_probs, random_sample):
    """

    Args:
        means: [batch, nmix, nc, H, W]
        logscales: [batch, nmix, nc, H, W]
        logit_probs:, [batch, nmix, H, W]
        coeffs: [batch, nmix, nc, H, W]
        random_sample: boolean

    Returns:
        samples [batch, nc, H, W]

    """
    # [batch, 1, H, W] -> [batch, 1, 1, H, W]
    index = logit_probs.argmax(dim=1, keepdim=True).unsqueeze(2)
    one_hot = means.new_zeros(means.size()).scatter_(1, index, 1)
    # [batch, nc, H, W]
    means = (means * one_hot).sum(dim=1)
    logscales = (logscales * one_hot).sum(dim=1)

    x = means
    if random_sample:
        u = means.new_zeros(means.size()).uniform_(1e-5, 1 - 1e-5)
        x = x + logscales.exp() * (torch.log(u) - torch.log(1.0 - u))

    return x
