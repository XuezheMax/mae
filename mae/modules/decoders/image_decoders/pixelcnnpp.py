__author__ = 'max'

import math
from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from mae.modules.networks.auto_regressives.pixelcnnpp import PixelCNNPP
from mae.modules.decoders.image_decoders.color_image_decoder import ColorImageDecoder
from mae.modules.utils import sample_from_discretized_mix_logistic


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

        self.z_transform = nn.Sequential(
            # state [b, z_channels, 8, 8]
            nn.ConvTranspose2d(z_channels, z_channels // 2, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(z_channels // 2),
            nn.ELU(),
            # state [b, z_channels / 2, 16, 16]
            nn.ConvTranspose2d(z_channels // 2, z_channels // 4, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(z_channels // 4),
            nn.ELU(),
            # state [b, z_channels / 4, 32, 32]
            nn.Conv2d(z_channels // 4, h_channels, 1)
            # state [b, h_channels, 32, 32]
        )

        hidden_channels = 64
        self.core = PixelCNNPP(3, self.nc, hidden_channels, 4, h_channels, dropout=dropout)
        self.output = nn.Sequential(
            # state [64, 32, 32]
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ELU(),
            # state [64, 32, 32]
            nn.Conv2d(64, (self.nc * 3 + 1) * self.nmix, 1, bias=False)
        )

        self.reset_parameters()

        if ngpu > 1:
            self.z_transform = nn.DataParallel(self.z_transform, device_ids=list(range(ngpu)))
            self.core = nn.DataParallel(self.core, device_ids=list(range(ngpu)))
            self.output = nn.DataParallel(self.output, device_ids=list(range(ngpu)))

    def reset_parameters(self):
        # //////// z_transform //////////
        m = self.z_transform[0]
        assert isinstance(m, nn.ConvTranspose2d)
        nn.init.xavier_normal_(m.weight)

        m = self.z_transform[3]
        assert isinstance(m, nn.ConvTranspose2d)
        nn.init.xavier_normal_(m.weight)

        m = self.z_transform[1]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

        m = self.z_transform[4]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

        m = self.z_transform[6]
        assert isinstance(m, nn.Conv2d)
        nn.init.xavier_normal_(m.weight)

        # //////// core //////////
        m = self.output[0]
        assert isinstance(m, nn.Conv2d)
        nn.init.xavier_normal_(m.weight)

        m = self.output[3]
        assert isinstance(m, nn.Conv2d)
        nn.init.xavier_normal_(m.weight)

        m = self.output[1]
        assert isinstance(m, nn.BatchNorm2d)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

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
        return x

    def forward(self, x, z):
        h = self.z_transform(z)
        return self.output(self.core(x, h=h))

    @classmethod
    def from_params(cls, params: Dict) -> "PixelCNNPPDecoderColorImage32x32":
        return PixelCNNPPDecoderColorImage32x32(**params)


PixelCNNPPDecoderColorImage32x32.register("pixelcnn++_color_32x32")
