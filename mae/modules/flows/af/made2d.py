__author__ = 'max'

from overrides import overrides
import torch.nn as nn
import torch
from typing import Dict, Tuple, List

from mae.modules.flows.af.af import AF2d


class MADE2d(AF2d):
    """

    """