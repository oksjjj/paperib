# -*- coding: utf-8 -*-
import os
import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Legacy flat layout (pre-THOC alignment). Prefer data/{SMAP,MSL,SMD}/.
_LEGACY_PREFIX = 'processed'


def data_dir_for(dataset):
    """
    Prepared-data directory for a dataset name (THOC-aligned).

    ``SMAP`` / ``MSL`` → ``data/SMAP`` / ``data/MSL``
    ``machine-*``     → ``data/SMD``
    ``P*``            → ``data/paperib`` (paperib PLMN exports)
    Falls back to ``processed/`` if the new layout is missing.
    """
    if dataset in ('SMAP', 'MSL'):
        modern = os.path.join('data', dataset)
    elif str(dataset).startswith('machine'):
        modern = os.path.join('data', 'SMD')
    elif str(dataset).startswith('P'):
        modern = os.path.join('data', 'paperib')
    else:
        modern = _LEGACY_PREFIX

    train_name = f'{dataset}_train.pkl'
    if os.path.isfile(os.path.join(modern, train_name)):
        return modern
    if os.path.isfile(os.path.join(_LEGACY_PREFIX, train_name)):
        return _LEGACY_PREFIX
    return modern


def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    elif str(dataset).startswith('P'):
        meta_path = os.path.join(data_dir_for(dataset), f'{dataset}_meta.json')
        if os.path.isfile(meta_path):
            import json
            with open(meta_path, encoding='utf-8') as f:
                return int(json.load(f)['x_dim'])
        raise ValueError(
            f'unknown paperib dataset {dataset!r}: missing {meta_path}'
        )
    else:
        raise ValueError('unknown dataset ' + str(dataset))


def default_pot_level(dataset):
    """
    Paper Appendix B POT low quantile by dataset.

    SMAP 0.07, MSL 0.01,
    SMD machine-1-* 0.005, machine-2-* 0.0025, machine-3-* 0.0001.
    """
    name = str(dataset)
    if name == 'SMAP':
        return 0.07
    if name == 'MSL':
        return 0.01
    if name.startswith('machine-1-'):
        return 0.005
    if name.startswith('machine-2-'):
        return 0.0025
    if name.startswith('machine-3-'):
        return 0.0001
    if name.startswith('machine') or name.startswith('P'):
        return 0.005  # fallback for SMD / paperib PLMN
    raise ValueError('unknown dataset ' + name)


def save_z(z, filename='z'):
    """Save the sampled z in txt files."""
    for i in range(0, z.shape[1], 20):
        with open(filename + '_' + str(i) + '.txt', 'w') as file:
            for j in range(z.shape[0]):
                for k in range(z.shape[2]):
                    file.write('%f ' % (z[j][i][k]))
                file.write('\n')
    i = z.shape[1] - 1
    with open(filename + '_' + str(i) + '.txt', 'w') as file:
        for j in range(z.shape[0]):
            for k in range(z.shape[2]):
                file.write('%f ' % (z[j][i][k]))
            file.write('\n')


def _fmt_exp_float(value):
    """Compact float tag for experiment folder names (e.g. 1e-3, 5e-4, 5)."""
    v = float(value)
    if v == 0:
        return '0'
    if abs(v) >= 1 and float(int(v)) == v:
        return str(int(v))
    s = f'{v:.0e}'.replace('+', '')
    if 'e-' in s:
        base, exp = s.split('e-')
        s = f'{base}e-{int(exp)}'
    elif 'e' in s:
        base, exp = s.split('e')
        s = f'{base}e{int(exp)}'
    return s


def build_experiment_name(config, run_name=None):
    """
    Stable slug from training settings so runs do not overwrite each other.

    Example: ``prior_es_nf``, ``noprior_es_nf_lr5e-4_eps1e-3_gclip5``.
    Non-paper training knobs are appended only when they differ from defaults.
    """
    if run_name:
        return str(run_name)

    prior = 'prior' if getattr(config, 'include_prior_in_loss', True) else 'noprior'
    es = 'es' if getattr(config, 'early_stop', False) else 'noes'
    pft = getattr(config, 'posterior_flow_type', 'nf')
    flow = 'nf' if pft == 'nf' else 'nonf'
    parts = [prior, es, flow]
    if not getattr(config, 'use_connected_z_p', True):
        parts.append('noz_p')
    if not getattr(config, 'use_connected_z_q', True):
        parts.append('noz_q')

    # Paper defaults: lr=1e-3, std_epsilon=1e-4, per-tensor clip=10
    lr = float(getattr(config, 'initial_lr', 1e-3))
    if abs(lr - 1e-3) > 1e-15:
        parts.append(f'lr{_fmt_exp_float(lr)}')
    eps = float(getattr(config, 'std_epsilon', 1e-4))
    if abs(eps - 1e-4) > 1e-15:
        parts.append(f'eps{_fmt_exp_float(eps)}')
    clip_mode = getattr(config, 'grad_clip_mode', 'per_tensor')
    clip_norm = float(getattr(config, 'gradient_clip_norm', 10.0))
    if clip_mode == 'global':
        parts.append(f'gclip{_fmt_exp_float(clip_norm)}')
    elif abs(clip_norm - 10.0) > 1e-15:
        parts.append(f'pclip{_fmt_exp_float(clip_norm)}')
    return '_'.join(parts)


def resolve_output_dirs(config, run_name=None,
                        save_root=None, result_root=None, log_root=None):
    """
    Nest outputs under ``{root}/{dataset}/{experiment_name}/``.

    Roots default to the current ``save_dir`` / ``result_dir`` / ``log_dir``
    (or ``model`` / ``result`` / ``log``). Explicit roots win when provided.
    """
    exp = build_experiment_name(config, run_name=run_name)
    dataset = config.dataset
    config.experiment_name = exp

    save_root = save_root if save_root is not None else (config.save_dir or 'model')
    result_root = result_root if result_root is not None else (config.result_dir or 'result')
    log_root = log_root if log_root is not None else (config.log_dir or 'log')

    config.save_dir = os.path.join(save_root, dataset, exp)
    config.result_dir = os.path.join(result_root, dataset, exp)
    config.log_dir = os.path.join(log_root, dataset, exp)
    return exp


def get_data(dataset, max_train_size=None, max_test_size=None, print_log=True,
             do_preprocess=True, train_start=0, test_start=0):
    """
    Load data from pkl files.

    Returns:
        ((train_data, None), (test_data, test_label))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size

    if print_log:
        print('load data of:', dataset)
        print("train: ", train_start, train_end)
        print("test: ", test_start, test_end)

    x_dim = get_data_dim(dataset)
    prefix = data_dir_for(dataset)
    with open(os.path.join(prefix, dataset + '_train.pkl'), "rb") as f:
        train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]

    try:
        with open(os.path.join(prefix, dataset + '_test.pkl'), "rb") as f:
            test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
    except (KeyError, FileNotFoundError):
        test_data = None

    try:
        with open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb") as f:
            test_label = pickle.load(f).reshape((-1,))[test_start:test_end]
    except (KeyError, FileNotFoundError):
        test_label = None

    if do_preprocess:
        train_data = preprocess(train_data)
        if test_data is not None:
            test_data = preprocess(test_data)

    if print_log:
        print("train set shape: ", train_data.shape)
        print("test set shape: ", test_data.shape)
        if test_label is not None:
            print("test set label shape: ", test_label.shape)

    return (train_data, None), (test_data, test_label)


def preprocess(df):
    """Return MinMax-normalized data."""
    df = np.asarray(df, dtype=np.float32)

    if df.ndim == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(np.isnan(df)):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num(df)

    df = MinMaxScaler().fit_transform(df)
    print('Data normalized')
    return df


def minibatch_slices_iterator(length, batch_size, ignore_incomplete_batch=False):
    start = 0
    stop1 = (length // batch_size) * batch_size
    while start < stop1:
        yield slice(start, start + batch_size, 1)
        start += batch_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class BatchSlidingWindow(object):
    """Mini-batch iterator for sliding windows."""

    def __init__(self, array_size, window_size, batch_size, excludes=None,
                 shuffle=False, ignore_incomplete_batch=False):
        if window_size < 1:
            raise ValueError('`window_size` must be at least 1')
        if array_size < window_size:
            raise ValueError('`array_size` must be at least as large as `window_size`')

        if excludes is not None:
            excludes = np.asarray(excludes, dtype=bool)
            if excludes.shape != (array_size,):
                raise ValueError(
                    f'The shape of `excludes` is expected to be {(array_size,)}, '
                    f'but got {excludes.shape}'
                )

        if excludes is not None:
            mask = np.logical_not(excludes)
        else:
            mask = np.ones([array_size], dtype=bool)
        mask[: window_size - 1] = False

        if excludes is not None:
            where_excludes = np.where(excludes)[0]
            for k in range(1, window_size):
                also_excludes = where_excludes + k
                also_excludes = also_excludes[also_excludes < array_size]
                mask[also_excludes] = False

        indices = np.arange(array_size)[mask]
        self._indices = indices.reshape([-1, 1])
        self._offsets = np.arange(-window_size + 1, 1)
        self._array_size = array_size
        self._window_size = window_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def get_iterator(self, arrays):
        arrays = tuple(np.asarray(a) for a in arrays)
        if not arrays:
            raise ValueError('`arrays` must not be empty')

        if self._shuffle:
            np.random.shuffle(self._indices)

        for s in minibatch_slices_iterator(
                length=len(self._indices),
                batch_size=self._batch_size,
                ignore_incomplete_batch=self._ignore_incomplete_batch):
            idx = self._indices[s] + self._offsets
            yield tuple(a[idx] if a.ndim == 1 else a[idx, :] for a in arrays)
