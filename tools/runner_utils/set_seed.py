import torch
import random
import numpy as np

from transformers.utils import (
    is_tf_available,
    is_torch_available,
)

def set_seed(seed_value: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    if is_torch_available():
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
