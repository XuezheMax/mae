__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from mae.modules.networks.weight_norm import LinearWeightNorm, Conv2dWeightNorm
from mae.modules.networks.auto_regressives.pixelcnn import PixelCNN
from mae.modules.decoders.image_decoders.binary_image_decoder import BinaryImageDecoder


class ReShape(nn.Module):
    def __init__(self, out_shape):
        super(ReShape, self).__init__()
        self.out_shape = out_shape

    def forward(self, input):
        return input.view(input.size(0), *self.out_shape)


class PixelCNNDecoderBinaryImage28x28(BinaryImageDecoder):
    """
    PixelCNN Deocder for binary image of 28x28 resolution.
    See paper https://arxiv.org/abs/1601.06759 and https://arxiv.org/abs/1611.02731.
    """

    def __init__(self, nz, mode, ngpu=1):
        super(PixelCNNDecoderBinaryImage28x28, self).__init__(nz, ngpu=ngpu)
        self.nc = 1
        self.fm_latent = 4
        self.img_latent = 28 * 28 * self.fm_latent
        self.z_transform = nn.Sequential(
            LinearWeightNorm(nz, self.img_latent),
            ReShape((self.fm_latent, 28, 28)),
            nn.ELU(),
        )

        if mode == 'small':
            kernal_sizes = [7, 7, 7, 5, 5, 3, 3]
        elif mode == 'large':
            kernal_sizes = [7, 7, 7, 7, 7, 5, 5, 5, 5, 3, 3, 3, 3]
        else:
            raise ValueError('unknown mode: %s' % mode)

        hidden_channels = 64
        self.core = nn.Sequential(
            PixelCNN(self.nc + self.fm_latent, hidden_channels, len(kernal_sizes), kernal_sizes, self.nc),
            Conv2dWeightNorm(hidden_channels, hidden_channels, 1, bias=True),
            nn.ELU(),
            Conv2dWeightNorm(hidden_channels, self.nc, 1, bias=True),
            nn.Sigmoid(),
        )

        if ngpu > 1:
            self.z_transform = nn.DataParallel(self.z_transform, device_ids=list(range(ngpu)))
            self.core = nn.DataParallel(self.core, device_ids=list(range(ngpu)))

    @overrides
    def z_shape(self) -> Tuple:
        return self.nz,

    @overrides
    def output_size(self) -> Tuple:
        """

        Returns: a tuple of the output shape of decoded x (excluding batch_size)

        """
        return self.nc, 28, 28

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
        H = W = 28
        batch_size, nz = z.size()

        # [batch, nz] --> [batch, fm, H, W]
        z = self.z_transform(z)
        img = z.new_zeros(batch_size, self.nc, H, W)
        # [batch, nc+fm, H, W]
        img = torch.cat([img, z], dim=1)
        for i in range(H):
            for j in range(W):
                # [batch, nc, H, W]
                recon_img = self.core(img)
                # [batch, nc]
                img[:, :self.nc, i, j] = torch.bernoulli(recon_img[:, :, i, j]) if random_sample else recon_img[:, :, i, j].ge(0.5).float()

        # [batch, nc, H, W]
        img_probs = self.core(img)
        return img[:, :self.nc], img_probs

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
        # [batch, nsamples, nz] --> [batch*nsamples, nz] --> [batch*nsample, fm, H, W]
        z = self.z_transform(z.view(batch_size * nsampels, *z_size[2:]))
        # [batch*nsample, fm, H, W] --> [batch, nsample, fm, H, W]
        z = z.view(batch_size, nsampels, *z.size()[1:])

        # [batch, nc, H, W] --> [batch, 1, nc, H, W] --> [batch, nsample, nc, H, W]
        img = x.unsqueeze(1).expand(batch_size, nsampels, *x.size()[1:])
        # [batch, nsample, nc+fm, H, W] --> [batch * nsamples, nc+fm, H, W]
        img = torch.cat([img, z], dim=2)
        img = img.view(-1, *img.size()[2:])

        # [batch * nsamples, *] --> [batch, nsamples, -1]
        recon_x = self.core(img).view(batch_size, nsampels, -1)
        # [batch, -1]
        x_flat = x.view(batch_size, -1)
        BCE = (recon_x + eps).log() * x_flat.unsqueeze(1) + (1.0 - recon_x + eps).log() * (1. - x_flat).unsqueeze(1)
        # [batch, nsamples]
        return BCE.sum(dim=2) * -1.0

    @overrides
    def initialize(self, x, z, init_scale=1.0):
        z_transform = self.z_transform.module if isinstance(self.z_transform, nn.DataParallel) else self.z_transform
        core = self.core.module if isinstance(self.core, nn.DataParallel) else self.core
        assert len(z_transform) == 3
        assert len(core) == 5

        z = z_transform[0].initialize(z, init_scale=init_scale)
        z = z_transform[2](z_transform[1](z))

        img = torch.cat([x, z], dim=1)
        img = core[0].initialize(img, init_scale=init_scale)
        img = core[1].initialize(img, init_scale=init_scale)
        img = core[2](img)
        img = core[3].initialize(img, init_scale=init_scale)
        return core[4](img)

    @classmethod
    def from_params(cls, params: Dict) -> "PixelCNNDecoderBinaryImage28x28":
        return PixelCNNDecoderBinaryImage28x28(**params)


PixelCNNDecoderBinaryImage28x28.register('pixelcnn_binary_28x28')
