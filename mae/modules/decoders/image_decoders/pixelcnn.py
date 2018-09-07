__author__ = 'max'

import math
from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

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
            nn.Linear(nz, self.img_latent),
            ReShape((self.fm_latent, 28, 28)),
            nn.BatchNorm2d(self.fm_latent),
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
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ELU(),
            nn.Conv2d(hidden_channels, self.nc, 1, bias=False),
            nn.Sigmoid(),
        )
        self.reset_parameters()

        if ngpu > 1:
            self.z_transform = nn.DataParallel(self.z_transform, device_ids=list(range(ngpu)))
            self.core = nn.DataParallel(self.core, device_ids=list(range(ngpu)))

    def reset_parameters(self):
        m = self.z_transform[0]
        assert isinstance(m, nn.Linear)
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

        m = self.z_transform[2]
        assert isinstance(m, nn.BatchNorm2d)
        m.weight.data.fill_(1)
        m.bias.data.zero_()

        m = self.core[1]
        assert isinstance(m, nn.Conv2d)
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        nn.init.normal_(m.weight, 0, math.sqrt(2. / n))

        m = self.core[4]
        assert isinstance(m, nn.Conv2d)
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        nn.init.normal_(m.weight, 0, math.sqrt(2. / n))

        m = self.core[2]
        assert isinstance(m, nn.BatchNorm2d)
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    @overrides
    def z_shape(self) -> Tuple:
        return self.nz,

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
        z = z.view(batch_size, nsampels, *z.size()[2:])

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

    @classmethod
    def from_params(cls, params: Dict) -> "PixelCNNDecoderBinaryImage28x28":
        return PixelCNNDecoderBinaryImage28x28(**params)


PixelCNNDecoderBinaryImage28x28.register('pixelcnn_binary_28x28')
