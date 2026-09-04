# -*- coding: utf-8 -*-
"""
OmniAnomaly entry point — PyTorch port of the official TensorFlow main.py
https://github.com/NetManAIOps/OmniAnomaly
"""
import argparse
import json
import logging
import os
import pickle
import time
import warnings

import numpy as np
import torch

from omni_anomaly.checkpoint import load_checkpoint
from omni_anomaly.device import get_device
from omni_anomaly.eval_methods import bf_search, calc_rank_metrics, pot_eval
from omni_anomaly.model import OmniAnomaly
from omni_anomaly.prediction import Predictor
from omni_anomaly.train_logger import experiment_logging
from omni_anomaly.training import Trainer
from omni_anomaly.utils import (
    default_pot_level,
    get_data,
    get_data_dim,
    resolve_output_dirs,
    save_z,
)


class ExpConfig:
    """Experiment configuration (defaults match the official ExpConfig)."""

    # dataset configuration
    dataset = "SMAP"
    x_dim = 38

    # model architecture configuration
    use_connected_z_q = True
    use_connected_z_p = True
    # True: ELBO with log p(z); False: official TF loss (reconstruction only)
    include_prior_in_loss = True

    # model parameters
    z_dim = 3
    rnn_cell = 'GRU'  # 'GRU', 'LSTM' or 'Basic'
    rnn_num_hidden = 500
    window_length = 100
    dense_dim = 500
    posterior_flow_type = 'nf'  # 'nf' or None
    nf_layers = 20
    max_epoch = 20  # paper: 20 epochs; best valid weights restored at end
    train_start = 0
    max_train_size = None
    batch_size = 50
    l2_reg = 0.0001  # paper: L2 coefficient 1e-4 on all layers
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4

    # evaluation parameters
    test_n_z = 1
    test_batch_size = 50
    test_start = 0
    max_test_size = None

    bf_search_min = -400.
    bf_search_max = 400.
    bf_search_step_size = 1.

    valid_step_freq = 100
    gradient_clip_norm = 10.
    # 'per_tensor': paper/TF-style clip_by_norm per gradient tensor
    # 'global': clip the whole-model grad norm (more stable late training)
    grad_clip_mode = 'per_tensor'

    early_stop = False             # train full max_epoch; restore best valid weights
    early_stop_patience = 30       # consecutive valid checks without improvement
    early_stop_min_epochs = 3      # do not stop before this many epochs
    early_stop_warmup_steps = 300  # ignore patience counting during warmup


    # pot parameters (paper Appendix B)
    # q = 1e-4 for all datasets
    # level is auto-set from dataset via default_pot_level() unless --level is given
    level = None
    pot_q = 1e-4

    # outputs config
    # Actual paths are resolved to {root}/{dataset}/{experiment_name}/
    save_z = False
    get_score_on_dim = False
    save_dir = 'model'
    restore_dir = None
    result_dir = 'result'
    train_score_filename = 'train_score.pkl'
    test_score_filename = 'test_score.pkl'
    experiment_name = None  # set by resolve_output_dirs
    run_name = None         # optional override for experiment_name

    # PyTorch-only
    log_dir = 'log'
    device = None  # auto: mps > cuda > cpu
    tensorboard = True

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k in dir(self)
            if not k.startswith('_') and not callable(getattr(self, k))
        }

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)


def parse_args():
    parser = argparse.ArgumentParser(
        description='OmniAnomaly (PyTorch port of NetManAIOps/OmniAnomaly)',
    )
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--max_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--initial_lr', type=float, default=None,
                        help='Adam learning rate (paper default: 1e-3)')
    parser.add_argument('--std_epsilon', type=float, default=None,
                        help='Floor added to softplus std (paper default: 1e-4). '
                             'Try 1e-3 to reduce decoder variance collapse.')
    parser.add_argument('--gradient_clip_norm', type=float, default=None,
                        help='Grad clip max-norm (paper default: 10)')
    parser.add_argument('--grad_clip_mode', type=str, default=None,
                        choices=('per_tensor', 'global'),
                        help="'per_tensor' = paper/TF (default); "
                             "'global' = clip whole-model grad norm")
    parser.add_argument('--stable_train', action='store_true',
                        help='Convenience preset: std_epsilon=1e-3, '
                             'initial_lr=5e-4, grad_clip_mode=global, '
                             'gradient_clip_norm=5 (does not override '
                             'explicit flags)')
    parser.add_argument('--z_dim', type=int, default=None)
    parser.add_argument('--window_length', type=int, default=None)
    parser.add_argument('--level', type=float, default=None,
                        help='POT low quantile (default: auto by dataset; '
                             'SMAP 0.07, MSL 0.01, SMD 0.005/0.0025/0.0001)')
    parser.add_argument('--pot_q', type=float, default=None,
                        help='POT risk q (paper: 1e-4)')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--restore_dir', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default=None,
                        help='mps / cuda / cpu (default: auto)')
    parser.add_argument('--valid_step_freq', type=int, default=None)
    parser.add_argument('--early_stop', action='store_true',
                        help='Enable patience-based early stopping '
                             '(default: run all max_epoch, then restore best)')
    parser.add_argument('--early_stop_patience', type=int, default=None,
                        help='Valid checks without improvement before stop')
    parser.add_argument('--early_stop_min_epochs', type=int, default=None)
    parser.add_argument('--early_stop_warmup_steps', type=int, default=None)
    parser.add_argument('--posterior_flow_type', type=str, default=None,
                        help="'nf' or 'none'")
    parser.add_argument('--no_tensorboard', action='store_true',
                        help='Disable TensorBoard logging')
    parser.add_argument('--exclude_prior', action='store_true',
                        help='Omit log p(z) from SGVB loss '
                             '(official TF ablation; default keeps prior)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Override auto experiment folder name '
                             '(default: prior/noprior_es/noes_nf/...)')
    return parser.parse_args()


def get_checkpoint_dir(config):
    """Checkpoint directory (already ``model/{dataset}/{experiment_name}``)."""
    return config.save_dir or 'model'


def _fmt(value, digits=4):
    """Format metric values for readable log output."""
    if value is None:
        return '-'
    try:
        if isinstance(value, (float, np.floating)):
            return f'{float(value):.{digits}f}'
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        # numpy scalars / strings that look numeric
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return str(value)


def print_metrics_summary(metrics):
    """
    Print evaluation results in a readable layout.

    All classification metrics use point adjustment (PA), consistent with
    the official OmniAnomaly evaluation protocol.
    """
    has_pot = 'pot-f1' in metrics
    has_bf = 'best-f1' in metrics
    pa_note = '포인트 보정 후'

    print()
    print('=' * 60)
    print(' EVALUATION SUMMARY')
    print(f' ({pa_note} — all classification metrics)')
    print('=' * 60)

    if has_pot:
        print()
        print(f'[POT]  {pa_note} | primary (threshold from train scores only)')
        print('-' * 60)
        print(f'  F1          {_fmt(metrics.get("pot-f1"))}')
        print(f'  Precision   {_fmt(metrics.get("pot-precision"))}')
        print(f'  Recall      {_fmt(metrics.get("pot-recall"))}')
        print(f'  TP / FP     {_fmt(metrics.get("pot-TP"), 0)} / {_fmt(metrics.get("pot-FP"), 0)}')
        print(f'  TN / FN     {_fmt(metrics.get("pot-TN"), 0)} / {_fmt(metrics.get("pot-FN"), 0)}')
        print(f'  Threshold   {_fmt(metrics.get("pot-threshold"), 6)}')
        print(f'  Latency     {_fmt(metrics.get("pot-latency"))}')
    else:
        print()
        print('[POT]  (skipped — no test labels)')

    if has_bf:
        print()
        print(f'[best-F1]  {pa_note} | oracle (threshold fit on test labels)')
        print('-' * 60)
        print(f'  F1          {_fmt(metrics.get("best-f1"))}')
        print(f'  Precision   {_fmt(metrics.get("precision"))}')
        print(f'  Recall      {_fmt(metrics.get("recall"))}')
        print(f'  TP / FP     {_fmt(metrics.get("TP"), 0)} / {_fmt(metrics.get("FP"), 0)}')
        print(f'  TN / FN     {_fmt(metrics.get("TN"), 0)} / {_fmt(metrics.get("FN"), 0)}')
        print(f'  Threshold   {_fmt(metrics.get("threshold"), 6)}')
        print(f'  Latency     {_fmt(metrics.get("latency"))}')

    if 'auroc' in metrics or 'auprc' in metrics:
        print()
        print(f'[Ranking]  {pa_note} | AUROC / AUPRC (PA at each threshold)')
        print('-' * 60)
        print(f'  AUROC       {_fmt(metrics.get("auroc"))}')
        print(f'  AUPRC       {_fmt(metrics.get("auprc"))}')
        if metrics.get('roc_curve'):
            print(f'  ROC image   {metrics["roc_curve"]}')
        if metrics.get('pr_curve'):
            print(f'  PR image    {metrics["pr_curve"]}')
        if metrics.get('roc_pr_combined'):
            print(f'  Combined    {metrics["roc_pr_combined"]}')

    print()
    print('[Training]')
    print('-' * 60)
    print(f'  best valid loss   {_fmt(metrics.get("best_valid_loss"))}')
    print(f'  stopped epoch     {_fmt(metrics.get("stopped_epoch"), 0)}')
    print(f'  stopped step      {_fmt(metrics.get("stopped_step"), 0)}')
    if metrics.get('train_time') is not None:
        print(f'  train time/epoch  {_fmt(metrics.get("train_time"))}s')
    if metrics.get('pred_total_time') is not None:
        print(f'  pred total time   {_fmt(metrics.get("pred_total_time"))}s')
    if metrics.get('early_stopped') is not None:
        print(f'  early stopped      {metrics.get("early_stopped")}')
    if metrics.get('best_checkpoint'):
        print(f'  best checkpoint   {metrics["best_checkpoint"]}')
    if metrics.get('tensorboard_dir'):
        print(f'  tensorboard       {metrics["tensorboard_dir"]}')
    print('=' * 60)
    print()


def run_experiment(config, device, log):
    os.makedirs(config.result_dir, exist_ok=True)
    if config.save_dir is not None:
        os.makedirs(get_checkpoint_dir(config), exist_ok=True)

    print('=' * 30 + ' Configurations ' + '=' * 30)
    print(json.dumps(config.to_dict(), indent=2, default=str))
    print(f'Using device: {device}')
    print(f'POT level={config.level}, q={config.pot_q}')
    print(f'include_prior_in_loss={config.include_prior_in_loss}')
    print(f'experiment_name={config.experiment_name}')
    print(f'save_dir={config.save_dir}')
    print(f'result_dir={config.result_dir}')
    print(f'log_dir={config.log_dir}')

    with open(os.path.join(config.result_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    (x_train, _), (x_test, y_test) = get_data(
        config.dataset,
        config.max_train_size,
        config.max_test_size,
        train_start=config.train_start,
        test_start=config.test_start,
    )

    model = OmniAnomaly(config).to(device)
    checkpoint_dir = get_checkpoint_dir(config)
    log_dir = config.log_dir or 'log'

    trainer = Trainer(
        model=model,
        device=device,
        max_epoch=config.max_epoch,
        batch_size=config.batch_size,
        valid_batch_size=config.test_batch_size,
        initial_lr=config.initial_lr,
        lr_anneal_epochs=config.lr_anneal_epoch_freq,
        lr_anneal_factor=config.lr_anneal_factor,
        grad_clip_norm=config.gradient_clip_norm,
        grad_clip_mode=config.grad_clip_mode,
        valid_step_freq=config.valid_step_freq,
        early_stop=config.early_stop,
        patience=config.early_stop_patience,
        early_stop_min_epochs=config.early_stop_min_epochs,
        early_stop_warmup_steps=config.early_stop_warmup_steps,
        l2_reg=config.l2_reg,
        log_dir=log_dir,
        dataset=config.dataset,
        checkpoint_dir=checkpoint_dir if config.save_dir is not None else None,
        config=config.to_dict(),
        tensorboard=config.tensorboard,
    )

    predictor = Predictor(
        model, device=device,
        batch_size=config.batch_size,
        n_z=config.test_n_z,
        last_point_only=True,
    )

    if config.restore_dir is not None:
        checkpoint, path = load_checkpoint(model, config.restore_dir, device)
        print(f'Model restored from {path}')
        log.info('Restored checkpoint: %s', path)

    if config.max_epoch > 0:
        train_start = time.time()
        best_valid_metrics = trainer.fit(x_train)
        train_time = (time.time() - train_start) / config.max_epoch
        best_valid_metrics['train_time'] = train_time
    else:
        best_valid_metrics = {}

    # train scores for POT
    train_score, train_z, train_pred_speed = predictor.get_score(x_train)
    if config.train_score_filename is not None:
        with open(os.path.join(config.result_dir, config.train_score_filename),
                  'wb') as file:
            pickle.dump(train_score, file)
    if config.save_z:
        save_z(train_z, 'train_z')

    if x_test is not None:
        test_start = time.time()
        test_score, test_z, pred_speed = predictor.get_score(x_test)
        test_time = time.time() - test_start
        if config.save_z:
            save_z(test_z, 'test_z')
        best_valid_metrics.update({
            'pred_time': pred_speed,
            'pred_total_time': test_time,
        })
        if config.test_score_filename is not None:
            with open(os.path.join(config.result_dir, config.test_score_filename),
                      'wb') as file:
                pickle.dump(test_score, file)

        if y_test is not None and len(y_test) >= len(test_score):
            if config.get_score_on_dim:
                test_score = np.sum(test_score, axis=-1)
                train_score = np.sum(train_score, axis=-1)

            t, th = bf_search(
                test_score, y_test[-len(test_score):],
                start=config.bf_search_min,
                end=config.bf_search_max,
                step_num=int(abs(config.bf_search_max - config.bf_search_min) /
                             config.bf_search_step_size),
                display_freq=50,
            )
            pot_result = pot_eval(
                train_score, test_score,
                y_test[-len(test_score):],
                q=config.pot_q,
                level=config.level,
            )
            rank_metrics = calc_rank_metrics(
                test_score, y_test[-len(test_score):],
                save_dir=config.result_dir,
                dataset=config.dataset,
            )
            best_valid_metrics.update({
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
            })
            best_valid_metrics.update(pot_result)
            best_valid_metrics.update(rank_metrics)

    # final save (official always saves after training)
    if config.save_dir is not None:
        from omni_anomaly.checkpoint import save_checkpoint
        path = save_checkpoint(
            model, config.to_dict(), get_checkpoint_dir(config),
            filename='model.pt',
        )
        print(f'Model saved to {path}')

    metrics_path = os.path.join(config.result_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(best_valid_metrics, f, indent=2, default=str)
    print(f'Metrics saved to {metrics_path}')

    print_metrics_summary(best_valid_metrics)
    log.info('Experiment finished')


def main():
    args = parse_args()
    # Optional stability preset; explicit CLI flags still win.
    if getattr(args, 'stable_train', False):
        if args.std_epsilon is None:
            args.std_epsilon = 1e-3
        if args.initial_lr is None:
            args.initial_lr = 5e-4
        if args.grad_clip_mode is None:
            args.grad_clip_mode = 'global'
        if args.gradient_clip_norm is None:
            args.gradient_clip_norm = 5.0

    config = ExpConfig()
    config.update_from_args(args)
    if args.early_stop:
        config.early_stop = True
    if args.no_tensorboard:
        config.tensorboard = False
    if args.exclude_prior:
        config.include_prior_in_loss = False
    if args.posterior_flow_type is not None:
        pft = args.posterior_flow_type
        config.posterior_flow_type = None if pft.lower() in ('none', 'null') else pft
    config.x_dim = get_data_dim(config.dataset)
    if config.level is None:
        config.level = default_pot_level(config.dataset)

    resolve_output_dirs(config, run_name=args.run_name)
    # Eval-only: default restore to this experiment's checkpoint dir
    if config.max_epoch == 0 and config.restore_dir is None:
        config.restore_dir = get_checkpoint_dir(config)

    if config.device:
        device = torch.device(config.device)
    else:
        # auto: MPS (Apple Silicon) > CUDA > CPU
        device = get_device(prefer_mps=True)

    os.makedirs(config.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    with experiment_logging(
        config.log_dir, config.dataset,
        mode='eval' if config.max_epoch == 0 else 'train',
    ) as (log_path, log):
        print(f'Log file: {log_path}')
        log.info('Experiment started')
        log.info('experiment_name=%s', config.experiment_name)
        log.info('save_dir=%s result_dir=%s log_dir=%s',
                 config.save_dir, config.result_dir, config.log_dir)
        log.info('Device: %s', device)
        log.info('POT level=%s (q=%s)', config.level, config.pot_q)
        log.info('include_prior_in_loss=%s', config.include_prior_in_loss)
        log.info(
            'train knobs: lr=%s std_epsilon=%s grad_clip_mode=%s '
            'gradient_clip_norm=%s',
            config.initial_lr, config.std_epsilon,
            config.grad_clip_mode, config.gradient_clip_norm,
        )
        run_experiment(config, device, log)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        main()
