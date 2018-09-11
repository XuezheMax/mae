__author__ = 'max'

import math
from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from mae.modules.networks.auto_regressives.pixelcnnpp import PixelCNNPP
from mae.modules.decoders.image_decoders.color_image_decoder import ColorImageDecoder
from mae.modules.decoders.image_decoders.pixelcnn import ReShape


class PixelCNNPPDecoderColorImage32x32(ColorImageDecoder):
    """
    PixelCNN++ Deocder for color image of 32x32 resolution.
    See paper https://arxiv.org/abs/1701.05517
    """