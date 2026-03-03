
"""A generic training wrapper."""
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader











@dataclass
class NNDataSetting:
    use_cnn_features: bool
    print("use_cnn_features",use_cnn_features)
