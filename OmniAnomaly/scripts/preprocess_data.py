#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
원본 SMD / SMAP / MSL 데이터를 학습용 pickle 로 변환한다.

사용 예:
  python scripts/preprocess_data.py --dataset SMD
  python scripts/preprocess_data.py --dataset SMAP
  python scripts/preprocess_data.py --dataset MSL \\
      --raw_dir data/raw/nasa --output_dir data/MSL
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
from pickle import dump

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _save_pkl(path, data):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as file:
        dump(data, file)
    print(f'Saved {path} | shape={np.asarray(data).shape}')


def load_and_save(category, filename, dataset, dataset_folder, output_dir):
    temp = np.genfromtxt(
        os.path.join(dataset_folder, category, filename),
        dtype=np.float32,
        delimiter=',',
    )
    print(dataset, category, filename, temp.shape)
    out = os.path.join(output_dir, f'{dataset}_{category}.pkl')
    _save_pkl(out, temp)


def preprocess_smd(raw_dir, output_dir):
    file_list = os.listdir(os.path.join(raw_dir, 'train'))
    for filename in file_list:
        if filename.endswith('.txt'):
            name = filename.strip('.txt')
            load_and_save('train', filename, name, raw_dir, output_dir)
            load_and_save('test', filename, name, raw_dir, output_dir)
            load_and_save('test_label', filename, name, raw_dir, output_dir)


def preprocess_nasa(dataset, raw_dir, output_dir):
    with open(os.path.join(raw_dir, 'labeled_anomalies.csv'), 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        res = [row for row in csv_reader][1:]
    res = sorted(res, key=lambda k: k[0])
    label_folder = os.path.join(raw_dir, 'test_label')
    os.makedirs(label_folder, exist_ok=True)
    data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
    labels = []
    for row in data_info:
        anomalies = ast.literal_eval(row[2])
        length = int(row[-1])
        label = np.zeros([length], dtype=bool)
        for anomaly in anomalies:
            label[anomaly[0]:anomaly[1] + 1] = True
        labels.extend(label)
    labels = np.asarray(labels)
    print(dataset, 'test_label', labels.shape)
    _save_pkl(os.path.join(output_dir, f'{dataset}_test_label.pkl'), labels)

    for category in ('train', 'test'):
        data = []
        for row in data_info:
            filename = row[0]
            temp = np.load(os.path.join(raw_dir, category, filename + '.npy'))
            data.extend(temp)
        data = np.asarray(data)
        print(dataset, category, data.shape)
        _save_pkl(os.path.join(output_dir, f'{dataset}_{category}.pkl'), data)


def parse_args():
    p = argparse.ArgumentParser(description='Preprocess SMD / SMAP / MSL')
    p.add_argument('--dataset', required=True, choices=['SMD', 'SMAP', 'MSL'])
    p.add_argument(
        '--raw_dir', default=None,
        help='Raw data root (default: data/raw/ServerMachineDataset or data/raw/nasa)',
    )
    p.add_argument(
        '--output_dir', default=None,
        help='Output dir (default: data/SMD, data/SMAP, or data/MSL)',
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)

    if args.dataset == 'SMD':
        raw_dir = args.raw_dir or os.path.join(
            'data', 'raw', 'ServerMachineDataset',
        )
        output_dir = args.output_dir or os.path.join('data', 'SMD')
        preprocess_smd(raw_dir, output_dir)
    else:
        raw_dir = args.raw_dir or os.path.join('data', 'raw', 'nasa')
        output_dir = args.output_dir or os.path.join('data', args.dataset)
        preprocess_nasa(args.dataset, raw_dir, output_dir)

    print(f'\nDone. Prepared files under {output_dir}/')


if __name__ == '__main__':
    main()
