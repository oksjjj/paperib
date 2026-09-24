#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download SMAP / MSL dataset (telemanom) into ``data/SMAP/raw`` and ``data/MSL/raw``.

Shared by OmniAnomaly, AnomalyTransformer, etc.

The original S3 URL (s3-us-west-2.amazonaws.com/telemanom/data.zip) returns 403.
This script downloads from Hugging Face: appleparan/telemanom

Next:
  python scripts/preprocess_data.py --dataset SMAP
  python scripts/preprocess_data.py --dataset MSL
"""
import os
import shutil
import sys

from huggingface_hub import hf_hub_download, list_repo_files

_PAPERIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

REPO_ID = 'appleparan/telemanom'
# Shared telemanom layout mirrored under each dataset's raw/ for uniformity.
DATA_DIRS = (
    os.path.join('data', 'SMAP', 'raw'),
    os.path.join('data', 'MSL', 'raw'),
)


def download_smap_msl():
    os.chdir(_PAPERIB_ROOT)
    primary = DATA_DIRS[0]
    for d in DATA_DIRS:
        os.makedirs(os.path.join(d, 'train'), exist_ok=True)
        os.makedirs(os.path.join(d, 'test'), exist_ok=True)

    print('Downloading labeled_anomalies.csv ...')
    csv_path = hf_hub_download(
        repo_id=REPO_ID,
        filename='labeled_anomalies.csv',
        repo_type='dataset',
    )
    for d in DATA_DIRS:
        shutil.copy2(csv_path, os.path.join(d, 'labeled_anomalies.csv'))

    npy_files = [
        f for f in list_repo_files(REPO_ID, repo_type='dataset')
        if f.startswith('data/data/train/') and f.endswith('.npy')
        or f.startswith('data/data/test/') and f.endswith('.npy')
    ]
    print(f'Downloading {len(npy_files)} .npy files ...')

    for i, remote_path in enumerate(sorted(npy_files), 1):
        category = 'train' if '/train/' in remote_path else 'test'
        filename = os.path.basename(remote_path)
        primary_path = os.path.join(primary, category, filename)

        if os.path.exists(primary_path):
            print(f'[{i}/{len(npy_files)}] skip {category}/{filename}')
        else:
            cached = hf_hub_download(
                repo_id=REPO_ID, filename=remote_path, repo_type='dataset',
            )
            shutil.copy2(cached, primary_path)
            print(f'[{i}/{len(npy_files)}] {category}/{filename}')

        # Mirror into MSL/raw (same telemanom files).
        for d in DATA_DIRS[1:]:
            dest = os.path.join(d, category, filename)
            if not os.path.exists(dest):
                shutil.copy2(primary_path, dest)

    print(f'\nDone. Dataset ready at ./data/SMAP/raw and ./data/MSL/raw')
    print('Next: python scripts/preprocess_data.py --dataset SMAP')
    print('      python scripts/preprocess_data.py --dataset MSL')


if __name__ == '__main__':
    try:
        download_smap_msl()
    except Exception as e:
        print('Download failed:', e, file=sys.stderr)
        print('\nAlternative (Kaggle API key required):', file=sys.stderr)
        print('  pip install kaggle', file=sys.stderr)
        print(
            '  kaggle datasets download -d '
            'patrickfleith/nasa-anomaly-detection-dataset-smap-msl',
            file=sys.stderr,
        )
        sys.exit(1)
