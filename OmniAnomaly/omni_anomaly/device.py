# -*- coding: utf-8 -*-
import torch


def get_device(prefer_mps=True):
    """Return the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
