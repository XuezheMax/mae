__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import torch.nn as nn

from mae.modules.networks.weight_norm import Conv2dWeightNorm, ConvTranspose2dWeightNorm
from mae.modules.networks.auto_regressives.pixelcnnpp import PixelCNNPP
from mae.modules.decoders.image_decoders.color_image_decoder import ColorImageDecoder
from mae.modules.utils import sample_from_discretized_mix_logistic


class _PixelCNNPPCore(nn.Module):
    def __init__(self, nc, z_channels, h_channels, nmix, dropout=0.):
        super(_PixelCNNPPCore, self).__init__()
        self.z_transform = nn.Sequential(
            # state [b, z_channels, 8, 8]
            ConvTranspose2dWeightNorm(z_channels, z_channels // 2, 3, 2, 1, 1, bias=False),
            nn.ELU(),
            # state [b, z_channels / 2, 16, 16]
            ConvTranspose2dWeightNorm(z_channels // 2, z_channels // 4, 3, 2, 1, 1, bias=False),
            nn.ELU(),
            # state [b, z_channels / 4, 32, 32]
            Conv2dWeightNorm(z_channels // 4, h_channels, 1),
            nn.ELU(),
            # state [b, h_channels, 32, 32]
        )

        hidden_channels = 64
        num_resnets = 4
        self.core = PixelCNNPP(3, nc, hidden_channels, num_resnets, h_channels, dropout=dropout)
        self.output = nn.Sequential(
            # state [64, 32, 32]
            Conv2dWeightNorm(hidden_channels, hidden_channels, 1, bias=False),
            nn.ELU(),
            # state [64, 32, 32]
            Conv2dWeightNorm(64, (nc * 3 + 1) * nmix, 1, bias=False)
        )

    def forward(self, x, z):
        h = self.z_transform(z)
        return self.output(self.core(x, h=h))


class PixelCNNPPDecoderColorImage32x32(ColorImageDecoder):
    """
    PixelCNN++ Deocder for color image of 32x32 resolution.
    See paper https://arxiv.org/abs/1701.05517
    """

    def __init__(self, z_channels, h_channels, nmix, dropout=0., ngpu=1):
        """

        Args:
            z_channels: number of filters of the input latent variable z (the shape of z is [batch, 8, 8, z_channels])
            h_channels: number of filters of the transformed latent variable used as the conditioned vector h (the shape of h is [batch, 32, 32, h_channels]
            nmix: number of mixures of the dicretized logistic distribution
            dropout: droput rate
            ngpu: number of gpus to use
        """
        super(PixelCNNPPDecoderColorImage32x32, self).__init__(nmix, ngpu=ngpu)
        self.z_channels = z_channels
        self.H = 8
        self.W = 8

        self.core = _PixelCNNPPCore(self.nc, z_channels, h_channels, nmix, dropout=dropout)
        if ngpu > 1:
            self.core = nn.DataParallel(self.core, device_ids=list(range(ngpu)))

    @overrides
    def z_shape(self) -> Tuple:
        return self.z_channels, self.H, self.W

    @overrides
    def output_size(self) -> Tuple:
        """

        Returns: a tuple of the output shape of decoded x (excluding batch_size)

        """
        return self.nc, 32, 32

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
        H = W = 32
        x = z.new_zeros(z.size(0), self.nc, H, W)
        for i in range(H):
            for j in range(W):
                # [batch, mix, nc, H, W]
                mu, log_scale, logit_probs, coeffs = self.execute(z, x)
                # [batch, nc, H, W]
                new_x = sample_from_discretized_mix_logistic(mu, log_scale, coeffs, logit_probs, random_sample)
                x[:, :, i, j] = new_x[:, :, i, j]
        return x, None

    def forward(self, x, z):
        return self.core(x, z)

    @classmethod
    def from_params(cls, params: Dict) -> "PixelCNNPPDecoderColorImage32x32":
        return PixelCNNPPDecoderColorImage32x32(**params)


PixelCNNPPDecoderColorImage32x32.register("pixelcnn++_color_32x32")
