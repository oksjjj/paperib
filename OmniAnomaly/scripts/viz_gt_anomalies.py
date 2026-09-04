# -*- coding: utf-8 -*-
"""
GT anomaly segment viewer (OmniAnomaly paper Fig.1 style).

Saves every ground-truth anomaly segment as its own PNG, with normal
context of 10× segment length on both sides.

Modes:
  --gt_only   : GT bands only (no model scores; use before training)
  default     : overlay TP/FP/FN from saved test_score (no point adjustment)

  TP : GT anomaly & predicted      → red filled circles
  FP : normal & predicted          → red triangles
  FN : GT anomaly & not predicted  → blue open circles

Examples:
    python scripts/viz_gt_anomalies.py --dataset machine-1-1 --gt_only
    python scripts/viz_gt_anomalies.py --dataset SMAP --gt_only
    python scripts/viz_gt_anomalies.py --dataset SMAP --run_name prior_noes_nf_lr5e-4_eps1e-3_gclip5
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import matplotlib
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omni_anomaly.utils import get_data, resolve_output_dirs


def _configure_pyplot():
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    return plt


def _segments(mask):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts.tolist(), ends.tolist()))


class _PathConfig:
    def __init__(self, args):
        self.dataset = args.dataset
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.include_prior_in_loss = not args.exclude_prior
        self.early_stop = args.early_stop
        self.posterior_flow_type = args.posterior_flow_type
        self.use_connected_z_p = not args.no_connected_z_p
        self.use_connected_z_q = not args.no_connected_z_q
        self.experiment_name = None


def parse_args():
    p = argparse.ArgumentParser(
        description='Save GT anomaly plots (optionally with TP/FP/FN markers)',
    )
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--gt_only', action='store_true',
                   help='Plot GT segments only (no scores/metrics required)')
    p.add_argument('--context_mult', type=float, default=10.0,
                   help='Normal context length multiplier on each side (default: 10)')
    p.add_argument('--out_dir', type=str, default=None,
                   help='Output root (default: viz_gt if --gt_only else viz_pred)')
    p.add_argument('--save_dir', type=str, default='model')
    p.add_argument('--result_dir', type=str, default='result')
    p.add_argument('--log_dir', type=str, default='log')
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--exclude_prior', action='store_true',
                   help='Use noprior experiment paths (default: prior included)')
    p.add_argument('--early_stop', action='store_true',
                   help='Match experiment that used early stopping (default: noes)')
    p.add_argument('--posterior_flow_type', type=str, default='nf')
    p.add_argument('--no_connected_z_p', action='store_true')
    p.add_argument('--no_connected_z_q', action='store_true')
    p.add_argument('--threshold', type=float, default=None,
                   help='Override best-F1 threshold (default: metrics.json)')
    return p.parse_args()


def _load_pred(result_dir, threshold_override):
    score_path = os.path.join(result_dir, 'test_score.pkl')
    with open(score_path, 'rb') as f:
        score = pickle.load(f)
    score = np.asarray(score, dtype=float)
    if score.ndim > 1:
        score = score.sum(axis=-1)

    if threshold_override is not None:
        thr = float(threshold_override)
        thr_src = 'cli'
    else:
        metrics_path = os.path.join(result_dir, 'metrics.json')
        with open(metrics_path) as f:
            metrics = json.load(f)
        thr = float(metrics['threshold'])
        thr_src = metrics_path

    pred = score < thr
    return score, pred, thr, thr_src, score_path


def plot_gt_segment(x, y, pred, start, end, index, n_total, context_mult,
                    threshold, out_path, gt_only=False):
    plt = _configure_pyplot()

    seg_len = max(end - start, 1)
    pad = int(round(seg_len * context_mult))
    left = max(0, start - pad)
    right = min(len(y), end + pad)
    xs = np.arange(left, right)
    x_win = x[left:right]
    y_win = y[left:right]
    n_dims = x.shape[1]

    if gt_only:
        pred_win = None
        n_tp = n_fp = n_fn = 0
    else:
        pred_win = pred[left:right]
        tp_mask = y_win & pred_win
        fp_mask = (~y_win) & pred_win
        fn_mask = y_win & (~pred_win)
        n_tp = int(tp_mask.sum())
        n_fp = int(fp_mask.sum())
        n_fn = int(fn_mask.sum())

    row_h = 0.5
    fig_h = max(5.0, 0.8 + row_h * n_dims)
    fig, axes = plt.subplots(
        n_dims, 1, figsize=(12, fig_h), sharex=True,
        gridspec_kw={'hspace': 0.06},
    )
    if n_dims == 1:
        axes = [axes]

    pink = '#f7b6c2'
    green = '#c8e6c9'
    line_c = '#1f4e79'
    red = '#d32f2f'
    fn_c = '#1565c0'  # blue: missed GT (not a model prediction)

    if gt_only:
        title = (
            f'{n_total} GT anomalies  |  [{index}]  '
            f'anomaly=[{start}, {end}) len={seg_len}  '
            f'±{context_mult}×'
        )
    else:
        title = (
            f'{n_total} GT anomalies  |  [{index}]  '
            f'anomaly=[{start}, {end}) len={seg_len}  '
            f'±{context_mult}×  thr={threshold} (no PA)  '
            f'TP={n_tp} FP={n_fp} FN={n_fn}'
        )
    fig.suptitle(title, fontsize=9, y=0.995)

    for i in range(n_dims):
        ax = axes[i]
        if start > left:
            ax.axvspan(left, start, color=green, alpha=0.85, lw=0, zorder=0)
        if right > end:
            ax.axvspan(end, right, color=green, alpha=0.85, lw=0, zorder=0)
        for a, b in _segments(y_win):
            ax.axvspan(left + a, left + b, color=pink, alpha=0.9, lw=0, zorder=1)
        ax.axvline(start, color='#c62828', lw=0.8, ls=':', zorder=3)
        ax.axvline(end, color='#c62828', lw=0.8, ls=':', zorder=3)

        series = x_win[:, i].astype(float)
        ax.plot(xs, series, color=line_c, lw=1.0, zorder=2)

        if not gt_only:
            if n_tp:
                ax.scatter(
                    xs[tp_mask], series[tp_mask],
                    s=10, c=red, marker='o', linewidths=0, zorder=5, label='TP',
                )
            if n_fp:
                ax.scatter(
                    xs[fp_mask], series[fp_mask],
                    s=14, c=red, marker='^', linewidths=0, zorder=5, label='FP',
                )
            if n_fn:
                ax.scatter(
                    xs[fn_mask], series[fn_mask],
                    s=12, facecolors='none', edgecolors=fn_c, marker='o',
                    linewidths=0.9, zorder=5, label='FN',
                )

        lo, hi = float(series.min()), float(series.max())
        if hi - lo < 1e-12:
            ax.set_ylim(lo - 0.1, hi + 0.1)
        else:
            pad_y = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad_y, hi + pad_y)
        ax.set_ylabel(f'm{i}', fontsize=8, rotation=0, labelpad=16, va='center')
        ax.tick_params(labelbottom=(i == n_dims - 1), labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

    axes[-1].set_xlabel('timestamp', fontsize=9)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=green, edgecolor='none',
              label=f'normal (±{context_mult}×)'),
        Patch(facecolor=pink, edgecolor='none', label='GT anomaly'),
        Line2D([0], [0], color=line_c, lw=1.2, label='metric'),
    ]
    if not gt_only:
        handles.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor=red,
                   markersize=6, linestyle='None', label='TP (pred∩GT)'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor=red,
                   markersize=7, linestyle='None', label='FP (pred∩¬GT)'),
            Line2D([0], [0], marker='o', color=fn_c, markerfacecolor='w',
                   markersize=6, linestyle='None', label='FN (GT∩¬pred)'),
        ])
    axes[0].legend(
        handles=handles,
        loc='upper right', fontsize=6.5, framealpha=0.92,
        ncol=3 if not gt_only else 1,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    out_root = args.out_dir or ('viz_gt' if args.gt_only else 'viz_pred')

    (_, _), (x_test, y_test) = get_data(args.dataset, do_preprocess=True)
    if y_test is None:
        raise SystemExit(f'No test labels for dataset={args.dataset}')

    x = np.asarray(x_test, dtype=float)
    y = np.asarray(y_test).reshape(-1).astype(bool)
    if len(y) != len(x):
        raise SystemExit(
            f'Length mismatch: x_test={len(x)} y_test={len(y)}'
        )

    if args.gt_only:
        pred = None
        thr = None
        thr_src = score_path = None
        exp = 'gt'
        out_dir = os.path.join(out_root, args.dataset)
        prefix = 'gt'
        header = f' GT ANOMALY ONLY  ({args.dataset})'
    else:
        cfg = _PathConfig(args)
        pft = args.posterior_flow_type
        cfg.posterior_flow_type = (
            None if pft.lower() in ('none', 'null') else pft
        )
        exp = resolve_output_dirs(cfg, run_name=args.run_name)
        score, pred, thr, thr_src, score_path = _load_pred(
            cfg.result_dir, args.threshold,
        )
        # Align with score length (sliding window last-point scores)
        x = x[-len(score):]
        y = y[-len(score):]
        pred = np.asarray(pred, dtype=bool).reshape(-1)
        assert len(x) == len(y) == len(pred) == len(score)
        out_dir = os.path.join(out_root, args.dataset, exp)
        prefix = 'pred'
        header = f' GT ANOMALY + MODEL PRED  ({args.dataset} / {exp})'

    segs = _segments(y)
    n_total = len(segs)
    os.makedirs(out_dir, exist_ok=True)

    print()
    print('=' * 64)
    print(header)
    print('=' * 64)
    if not args.gt_only:
        print(f'  experiment           : {exp}')
        print(f'  result_dir           : {cfg.result_dir}')
        print(f'  test_score           : {score_path}')
        print(f'  threshold            : {thr}  ({thr_src})')
        print(f'  point adjustment     : False')
    print(f'  GT anomaly segments  : {n_total}')
    print(f'  GT anomaly points    : {int(y.sum())} / {len(y)}')
    print(f'  context multiplier   : ±{args.context_mult}×')
    print(f'  output dir           : {out_dir}')
    if not args.gt_only:
        tp_all = int((y & pred).sum())
        fp_all = int(((~y) & pred).sum())
        fn_all = int((y & (~pred)).sum())
        tn_all = int(((~y) & (~pred)).sum())
        print('-' * 64)
        print(f'  TP points (no PA)    : {tp_all}')
        print(f'  FP points (no PA)    : {fp_all}')
        print(f'  FN points (no PA)    : {fn_all}')
        print(f'  TN points (no PA)    : {tn_all}')
    print('-' * 64)
    show_n = min(50, n_total)
    for i, (a, b) in enumerate(segs[:show_n]):
        print(f'  [{i:4d}]  start={a:8d}  end={b:8d}  len={b - a:6d}')
    if n_total > show_n:
        print(f'  ... ({n_total - show_n} more)')
    print('=' * 64)
    print(f'\nSaving {n_total} figures...')

    for i, (start, end) in enumerate(segs):
        out_path = os.path.join(out_dir, f'{prefix}_{i:04d}.png')
        plot_gt_segment(
            x, y, pred, start, end, i, n_total, args.context_mult,
            thr, out_path, gt_only=args.gt_only,
        )
        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            print(f'  [{i + 1}/{n_total}] {out_path}')

    print(f'\nDone. {n_total} figures saved under {out_dir}/')


if __name__ == '__main__':
    main()
