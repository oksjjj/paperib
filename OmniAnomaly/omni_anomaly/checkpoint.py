# -*- coding: utf-8 -*-
import os

import torch

BEST_MODEL_NAME = 'best_model.pt'
LEGACY_MODEL_NAME = 'model.pt'


def get_checkpoint_path(restore_dir, prefer_best=True):
    if prefer_best:
        candidates = [BEST_MODEL_NAME, LEGACY_MODEL_NAME]
    else:
        candidates = [LEGACY_MODEL_NAME, BEST_MODEL_NAME]

    for name in candidates:
        path = os.path.join(restore_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f'No checkpoint found in {restore_dir} '
        f'(tried {BEST_MODEL_NAME}, {LEGACY_MODEL_NAME})'
    )


def save_checkpoint(model, config, save_dir, filename=BEST_MODEL_NAME, extra=None):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    payload = {
        'model_state_dict': model.state_dict(),
        'config': dict(config) if config is not None else {},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(model, restore_dir, device, prefer_best=True):
    path = get_checkpoint_path(restore_dir, prefer_best=prefer_best)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint, path
