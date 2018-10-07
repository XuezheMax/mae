__author__ = 'max'

from collections import OrderedDict
from overrides import overrides
from typing import Dict, Tuple
import torch.nn as nn

from mae.modules.networks.weight_norm import Conv2dWeightNorm, ConvTranspose2dWeightNorm
from mae.modules.networks.auto_regressives.pixelcnnpp import PixelCNNPP
from mae.modules.decoders.image_decoders.color_image_decoder import ColorImageDecoder
from mae.modules.utils import sample_from_discretized_mix_logistic


class _PixelCNNPPCore(nn.Module):
    def __init__(self, nc, z_channels, h_channels, hidden_channels, levels, num_resnets, nmix, dropout=0., activation='concat_elu'):
        super(_PixelCNNPPCore, self).__init__()
        in_channels = z_channels
        blocks = []
        for i in range(levels - 1):
            blocks.append(('level%d' % i, ConvTranspose2dWeightNorm(in_channels, in_channels // 2, 3, 2, 1, 1, bias=True)))
            blocks.append(('elu%d' %i, nn.ELU(inplace=True)))
            in_channels = in_channels // 2
        blocks.append(('h_level', Conv2dWeightNorm(in_channels, h_channels, 1)))
        self.z_transform = nn.Sequential(OrderedDict(blocks))

        self.core = PixelCNNPP(levels, nc, hidden_channels, num_resnets, h_channels, dropout=dropout, activation=activation)

        self.output = nn.Sequential(
            nn.ELU(inplace=True),
            # state [hidden_channels, H, W]
            Conv2dWeightNorm(hidden_channels, hidden_channels, 1, bias=True),
            nn.ELU(inplace=True),
            # state [hidden_channels * 2, H, W]
            Conv2dWeightNorm(hidden_channels, (nc * 3 + 1) * nmix, 1, bias=True)
            # state [10 * nmix, H, W]
        )

    def forward(self, x, z):
        h = self.z_transform(z)
        return self.output(self.core(x, h=h))

    def initialize(self, x, z, init_scale=1.0):
        h = z
        for layer in self.z_transform:
            if isinstance(layer, nn.ELU):
                h = layer(h)
            else:
                h = layer.initialize(h, init_scale=init_scale)
        output = self.core.initialize(x, h=h, init_scale=init_scale)
        assert len(self.output) == 4
        output = self.output[1].initialize(self.output[0](output), init_scale=init_scale)
        output = self.output[3].initialize(self.output[2](output), init_scale=init_scale)
        return output


class PixelCNNPPDecoderColorImage32x32(ColorImageDecoder):
    """
    PixelCNN++ Deocder for color image of 32x32 resolution.
    See paper https://arxiv.org/abs/1701.05517
    """

    def __init__(self, z_channels, h_channels, nmix, dropout=0., activation='concat_elu', ngpu=1):
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
        self.image_size = 32

        levels = 3
        hidden_channels = 64
        num_resnets = 4
        self.core = _PixelCNNPPCore(self.nc, z_channels, h_channels, hidden_channels, levels, num_resnets, nmix, dropout=dropout, activation=activation)
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
        return self.nc, self.image_size, self.image_size

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
        H = W = self.image_size
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

    @overrides
    def initialize(self, x, z, init_scale=1.0):
        core = self.core.module if isinstance(self.core, nn.DataParallel) else self.core
        assert isinstance(core, _PixelCNNPPCore)
        return core.initialize(x, z, init_scale=init_scale)

    @classmethod
    def from_params(cls, params: Dict) -> "PixelCNNPPDecoderColorImage32x32":
        return PixelCNNPPDecoderColorImage32x32(**params)


class PixelCNNPPDecoderColorImage64x64(PixelCNNPPDecoderColorImage32x32):
    """
    PixelCNN++ Deocder for color image of 64x64 resolution.
    See paper https://arxiv.org/abs/1701.05517
    """

    def __init__(self, z_channels, h_channels, nmix, dropout=0., activation='concat_elu', ngpu=1):
        """

        Args:
            z_channels: number of filters of the input latent variable z (the shape of z is [batch, 8, 8, z_channels])
            h_channels: number of filters of the transformed latent variable used as the conditioned vector h (the shape of h is [batch, 64, 64, h_channels]
            nmix: number of mixures of the dicretized logistic distribution
            dropout: droput rate
            ngpu: number of gpus to use
        """
        super(PixelCNNPPDecoderColorImage64x64, self).__init__(z_channels, h_channels, nmix=nmix, dropout=dropout, activation=activation, ngpu=ngpu)
        self.image_size = 64

        levels = 4
        hidden_channels = 64
        num_resnets = 4
        self.core = _PixelCNNPPCore(self.nc, z_channels, h_channels, hidden_channels, levels, num_resnets, nmix, dropout=dropout, activation=activation)
        if ngpu > 1:
            self.core = nn.DataParallel(self.core, device_ids=list(range(ngpu)))

    @classmethod
    def from_params(cls, params: Dict) -> "PixelCNNPPDecoderColorImage64x64":
        return PixelCNNPPDecoderColorImage64x64(**params)


PixelCNNPPDecoderColorImage32x32.register("pixelcnn++_color_32x32")
PixelCNNPPDecoderColorImage64x64.register("pixelcnn++_color_64x64")
