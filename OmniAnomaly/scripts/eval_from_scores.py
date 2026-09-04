# -*- coding: utf-8 -*-
"""
Re-evaluate POT / best-F1 from saved score pickles (no re-training / re-scoring).

Example:
    python scripts/eval_from_scores.py --dataset SMAP
    python scripts/eval_from_scores.py --dataset SMAP --exclude_prior
    python scripts/eval_from_scores.py --dataset machine-1-1 --run_name noprior_noes_paper
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from main import print_metrics_summary
from omni_anomaly.eval_methods import bf_search, calc_rank_metrics, pot_eval
from omni_anomaly.train_logger import experiment_logging
from omni_anomaly.utils import default_pot_level, get_data, resolve_output_dirs


class _PathConfig:
    """Minimal config object for resolve_output_dirs."""

    def __init__(self, args):
        self.dataset = args.dataset
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.include_prior_in_loss = not args.exclude_prior
        self.early_stop = args.early_stop
        self.posterior_flow_type = (
            None if args.posterior_flow_type in ('none', 'null', None)
            else args.posterior_flow_type
        )
        self.use_connected_z_p = not args.no_connected_z_p
        self.use_connected_z_q = not args.no_connected_z_q
        self.experiment_name = None


def parse_args():
    p = argparse.ArgumentParser(description='Re-evaluate from saved scores')
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--save_dir', type=str, default='model')
    p.add_argument('--result_dir', type=str, default='result')
    p.add_argument('--log_dir', type=str, default='log')
    p.add_argument('--run_name', type=str, default=None,
                   help='Override auto experiment folder name')
    p.add_argument('--exclude_prior', action='store_true',
                   help='Match noprior run paths (default: prior included)')
    p.add_argument('--early_stop', action='store_true',
                   help='Match experiment that used early stopping (default: noes)')
    p.add_argument('--posterior_flow_type', type=str, default='nf',
                   help="'nf' or 'none' (for path resolution)")
    p.add_argument('--no_connected_z_p', action='store_true')
    p.add_argument('--no_connected_z_q', action='store_true')
    p.add_argument('--train_score', type=str, default='train_score.pkl')
    p.add_argument('--test_score', type=str, default='test_score.pkl')
    p.add_argument('--level', type=float, default=None,
                   help='POT low quantile (default: auto by dataset)')
    p.add_argument('--pot_q', type=float, default=1e-4,
                   help='POT risk q (paper: 1e-4)')
    p.add_argument('--bf_search_min', type=float, default=-400.)
    p.add_argument('--bf_search_max', type=float, default=400.)
    p.add_argument('--bf_search_step_size', type=float, default=1.)
    p.add_argument('--get_score_on_dim', action='store_true')
    return p.parse_args()


def run_eval(args, result_dir, log):
    level = args.level if args.level is not None else default_pot_level(args.dataset)
    train_path = os.path.join(result_dir, args.train_score)
    test_path = os.path.join(result_dir, args.test_score)

    with open(train_path, 'rb') as f:
        train_score = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_score = pickle.load(f)

    (_, _), (_, y_test) = get_data(args.dataset, do_preprocess=True)
    if y_test is None:
        raise RuntimeError(f'No test labels for dataset={args.dataset}')

    y_test = y_test[-len(test_score):]
    if args.get_score_on_dim:
        test_score = np.sum(test_score, axis=-1)
        train_score = np.sum(train_score, axis=-1)

    print(f'train_score: {train_score.shape}, test_score: {test_score.shape}')
    print(f'POT q={args.pot_q}, level={level}'
          f'{" (auto)" if args.level is None else " (manual)"}')
    log.info(
        'Re-eval dataset=%s q=%s level=%s train_score=%s test_score=%s',
        args.dataset, args.pot_q, level, train_path, test_path,
    )

    t, th = bf_search(
        test_score, y_test,
        start=args.bf_search_min,
        end=args.bf_search_max,
        step_num=int(abs(args.bf_search_max - args.bf_search_min) /
                     args.bf_search_step_size),
        display_freq=50,
    )
    pot_result = pot_eval(
        train_score, test_score, y_test,
        q=args.pot_q, level=level,
    )
    rank_metrics = calc_rank_metrics(
        test_score, y_test,
        save_dir=result_dir,
        dataset=args.dataset,
    )

    metrics = {
        'best-f1': t[0],
        'precision': t[1],
        'recall': t[2],
        'TP': t[3],
        'TN': t[4],
        'FP': t[5],
        'FN': t[6],
        'latency': t[-1],
        'threshold': th,
        'point_adjustment': True,
        'dataset': args.dataset,
        'pot_q': args.pot_q,
        'level': level,
        'level_source': 'manual' if args.level is not None else 'auto',
        'train_score': train_path,
        'test_score': test_path,
    }
    metrics.update(pot_result)
    metrics.update(rank_metrics)

    out_path = os.path.join(result_dir, 'metrics_reeval.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f'Metrics saved to {out_path}')
    print_metrics_summary(metrics)
    log.info('Re-eval finished. metrics=%s', out_path)
    return metrics


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)
    cfg = _PathConfig(args)
    if args.posterior_flow_type is not None:
        pft = args.posterior_flow_type
        cfg.posterior_flow_type = None if pft.lower() in ('none', 'null') else pft
    exp = resolve_output_dirs(cfg, run_name=args.run_name)

    print(f'experiment_name={exp}')
    print(f'result_dir={cfg.result_dir}')
    print(f'log_dir={cfg.log_dir}')

    with experiment_logging(cfg.log_dir, args.dataset, mode='eval') as (log_path, log):
        print(f'Log file: {log_path}')
        log.info('Score re-evaluation started')
        log.info('experiment_name=%s result_dir=%s', exp, cfg.result_dir)
        run_eval(args, cfg.result_dir, log)


if __name__ == '__main__':
    main()
