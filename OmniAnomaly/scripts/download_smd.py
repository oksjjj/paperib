#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download SMD (Server Machine Dataset) into ./data/raw/ServerMachineDataset/

Source: https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset

Next:
  python scripts/preprocess_data.py --dataset SMD
"""
import io
import os
import sys
import zipfile
from urllib.error import URLError
from urllib.request import urlopen

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

ZIP_URL = 'https://codeload.github.com/NetManAIOps/OmniAnomaly/zip/refs/heads/master'
ZIP_PREFIX = 'OmniAnomaly-master/ServerMachineDataset/'
OUTPUT_DIR = os.path.join('data', 'raw', 'ServerMachineDataset')


def download_smd(force=False):
    os.chdir(_REPO_ROOT)

    if os.path.isdir(OUTPUT_DIR) and os.listdir(OUTPUT_DIR) and not force:
        has_data = os.path.isdir(os.path.join(OUTPUT_DIR, 'train'))
        if has_data:
            print(f'{OUTPUT_DIR}/ already exists. Use --force to re-download.')
            return

    print('Downloading SMD from NetManAIOps/OmniAnomaly ...')
    print(f'URL: {ZIP_URL}')

    try:
        with urlopen(ZIP_URL, timeout=120) as response:
            data = response.read()
    except URLError as e:
        print(f'Download failed: {e}', file=sys.stderr)
        print('\nAlternative: git clone the dataset folder only', file=sys.stderr)
        print(
            '  git clone --depth 1 --filter=blob:none --sparse '
            'https://github.com/NetManAIOps/OmniAnomaly.git _omni_tmp',
            file=sys.stderr,
        )
        print(
            '  cd _omni_tmp && git sparse-checkout set ServerMachineDataset',
            file=sys.stderr,
        )
        print(
            f'  mkdir -p ../data/raw && mv ServerMachineDataset ../{OUTPUT_DIR} '
            '&& cd .. && rm -rf _omni_tmp',
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    extracted = 0

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = [
            n for n in zf.namelist()
            if n.startswith(ZIP_PREFIX) and not n.endswith('/')
        ]
        print(f'Extracting {len(members)} files to ./{OUTPUT_DIR}/')

        for name in members:
            rel_path = name[len(ZIP_PREFIX):]
            if not rel_path or rel_path == '.gitkeep':
                continue
            dest = os.path.join(OUTPUT_DIR, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(name) as src, open(dest, 'wb') as dst:
                dst.write(src.read())
            extracted += 1
            if extracted % 20 == 0 or extracted == len(members):
                print(f'  [{extracted}/{len(members)}] {rel_path}')

    train_count = len(os.listdir(os.path.join(OUTPUT_DIR, 'train')))
    print(f'\nDone. {train_count} machines in ./{OUTPUT_DIR}/')
    print('Next: python scripts/preprocess_data.py --dataset SMD')


if __name__ == '__main__':
    force = '--force' in sys.argv
    download_smd(force=force)
