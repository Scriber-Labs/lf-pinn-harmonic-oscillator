from __future__ import annotations
import random
import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy and PyTorch RNGs.

    Parameters
    ----------
    seed : int
        Integer seed.
    deterministic : bool, default = False
        If True, enables deterministic algorithms (slower, but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.maniual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
