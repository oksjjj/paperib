# -*- coding: utf-8 -*-
"""Trainer — PyTorch port of omni_anomaly.training.Trainer."""
import copy
import logging
import os
import time
from datetime import datetime

import numpy as np
import torch

from omni_anomaly.checkpoint import save_checkpoint
from omni_anomaly.train_logger import TrainHistory
from omni_anomaly.utils import BatchSlidingWindow

__all__ = ['Trainer']

logger = logging.getLogger('omni_anomaly.train')


def _clip_grad_norm_per_tensor(parameters, max_norm):
    """Match TF ``tf.clip_by_norm`` applied to each gradient tensor."""
    for p in parameters:
        if p.grad is not None:
            torch.nn.utils.clip_grad_norm_(p, max_norm)


def _clip_grad_norm_global(parameters, max_norm):
    """Clip the global (whole-model) gradient norm."""
    torch.nn.utils.clip_grad_norm_(
        [p for p in parameters if p.grad is not None],
        max_norm,
    )


class Trainer:
    """
    OmniAnomaly trainer.

    Always tracks the best validation loss and restores those weights at
    the end. When ``early_stop=True``, also stops early if validation loss
    does not improve for ``patience`` consecutive checks (after warmup /
    min epochs).
    """

    def __init__(
        self,
        model,
        device,
        n_z=None,
        max_epoch=256,
        max_step=None,
        batch_size=256,
        valid_batch_size=1024,
        valid_step_freq=100,
        initial_lr=0.001,
        lr_anneal_epochs=10,
        lr_anneal_factor=0.75,
        grad_clip_norm=10.0,
        grad_clip_mode='per_tensor',
        early_stop=False,
        patience=30,
        early_stop_min_epochs=3,
        early_stop_warmup_steps=300,
        l2_reg=0.0001,
        log_dir=None,
        dataset='default',
        checkpoint_dir=None,
        config=None,
        tensorboard=True,
    ):
        if max_epoch is None and max_step is None:
            raise ValueError('At least one of `max_epoch` and `max_step` '
                             'should be specified')
        if grad_clip_mode not in ('per_tensor', 'global'):
            raise ValueError(
                "grad_clip_mode must be 'per_tensor' or 'global', "
                f"got {grad_clip_mode!r}"
            )

        self.model = model
        self.device = device
        self.n_z = n_z
        self.max_epoch = max_epoch
        self.max_step = max_step
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.valid_step_freq = valid_step_freq
        self.initial_lr = initial_lr
        self.lr_anneal_epochs = lr_anneal_epochs
        self.lr_anneal_factor = lr_anneal_factor
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip_mode = grad_clip_mode
        self.early_stop = early_stop
        self.patience = patience
        self.early_stop_min_epochs = early_stop_min_epochs
        self.early_stop_warmup_steps = early_stop_warmup_steps
        self.l2_reg = l2_reg
        self.dataset = dataset
        self.history = TrainHistory(log_dir, dataset) if log_dir else None
        self.checkpoint_dir = checkpoint_dir
        self.config = config or {}
        self.best_checkpoint_path = None
        self.tb_dir = None
        self.tb_writer = None

        if log_dir and tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.tb_dir = os.path.join(log_dir, 'tensorboard', dataset, stamp)
            os.makedirs(self.tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(self.tb_dir)

        # Paper: L2 regularization coefficient 1e-4 on all layers
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=initial_lr, weight_decay=l2_reg,
        )
    def _log(self, message):
        logger.info(message)
        print(message)

    def _record_step(self, **kwargs):
        if self.history is not None:
            self.history.add(**kwargs)

    def _tb_scalar(self, tag, value, step):
        if self.tb_writer is None:
            return
        if value is None or not np.isfinite(value):
            return
        self.tb_writer.add_scalar(tag, value, step)

    def _eval_loss(self, data, batch_size):
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        sw = BatchSlidingWindow(
            array_size=len(data),
            window_size=self.model.window_length,
            batch_size=batch_size,
        )
        with torch.no_grad():
            for (batch_x,) in sw.get_iterator([data]):
                x = torch.from_numpy(batch_x).to(self.device)
                loss = self.model.get_training_loss(x, n_z=self.n_z)
                total_loss += loss.item() * len(batch_x)
                total_count += len(batch_x)
        self.model.train()
        return total_loss / max(total_count, 1)

    def _save_history(self):
        if self.history is None:
            return {}
        json_path, csv_path = self.history.save()
        paths = {}
        if json_path:
            paths['train_history_json'] = json_path
        if csv_path:
            paths['train_history_csv'] = csv_path
        return paths

    def _save_best_checkpoint(self, best_valid_loss, epoch, step):
        if not self.checkpoint_dir:
            return None
        path = save_checkpoint(
            self.model,
            self.config,
            self.checkpoint_dir,
            extra={
                'best_valid_loss': float(best_valid_loss),
                'epoch': int(epoch),
                'step': int(step),
                'dataset': self.dataset,
            },
        )
        self.best_checkpoint_path = path
        self._log(f'Best model saved to {path} (valid_loss={best_valid_loss:.6f})')
        return path

    def fit(self, values, valid_portion=0.3):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError('`values` must be a 2-D array')

        n = int(len(values) * valid_portion)
        train_values, valid_values = values[:-n], values[-n:]

        train_sw = BatchSlidingWindow(
            array_size=len(train_values),
            window_size=self.model.window_length,
            batch_size=self.batch_size,
            shuffle=True,
            ignore_incomplete_batch=True,
        )

        lr = self.initial_lr
        best_valid_loss = None
        best_state = None
        patience_counter = 0
        early_stopped = False
        train_batch_times = []
        valid_batch_times = []

        global_step = 0
        self._log(f'train_values: {train_values.shape}')
        if self.early_stop:
            self._log(
                f'Early stop: patience={self.patience} '
                f'(valid checks), min_epochs={self.early_stop_min_epochs}, '
                f'warmup_steps={self.early_stop_warmup_steps}'
            )
        if self.tb_dir:
            self._log(f'TensorBoard log dir: {self.tb_dir}')
            self._log(
                f'  view: tensorboard --logdir '
                f'{os.path.join(os.path.dirname(os.path.dirname(self.tb_dir)))}'
            )

        epoch = 0
        stop = False
        try:
            while not stop:
                epoch += 1
                if self.max_epoch is not None and epoch > self.max_epoch:
                    break

                self.model.train()
                epoch_start = time.time()

                for batch_x, in train_sw.get_iterator([train_values]):
                    if self.max_step is not None and global_step >= self.max_step:
                        stop = True
                        break

                    batch_start = time.time()
                    x = torch.from_numpy(batch_x).to(self.device)

                    self.optimizer.zero_grad()
                    loss = self.model.get_training_loss(x, n_z=self.n_z)
                    loss.backward()

                    if self.grad_clip_norm:
                        if self.grad_clip_mode == 'global':
                            _clip_grad_norm_global(
                                self.model.parameters(), self.grad_clip_norm,
                            )
                        else:
                            _clip_grad_norm_per_tensor(
                                self.model.parameters(), self.grad_clip_norm,
                            )

                    self.optimizer.step()
                    train_batch_times.append(time.time() - batch_start)
                    global_step += 1

                    self._tb_scalar(
                        'train/loss_step', float(loss.item()), global_step,
                    )

                    if global_step % self.valid_step_freq == 0:
                        valid_start = time.time()
                        valid_loss = self._eval_loss(
                            valid_values, self.valid_batch_size,
                        )
                        valid_batch_times.append(time.time() - valid_start)
                        train_duration = time.time() - epoch_start

                        track_stop = global_step > self.early_stop_warmup_steps
                        is_best = False
                        if track_stop:
                            is_best = (
                                best_valid_loss is None
                                or valid_loss < best_valid_loss
                            )
                            if is_best:
                                best_valid_loss = valid_loss
                                best_state = copy.deepcopy(
                                    self.model.state_dict(),
                                )
                                patience_counter = 0
                                self._save_best_checkpoint(
                                    best_valid_loss, epoch, global_step,
                                )
                            elif self.early_stop:
                                patience_counter += 1
                        else:
                            # Warmup: still track a provisional best for restore
                            if (
                                best_valid_loss is None
                                or valid_loss < best_valid_loss
                            ):
                                best_valid_loss = valid_loss
                                best_state = copy.deepcopy(
                                    self.model.state_dict(),
                                )
                                self._save_best_checkpoint(
                                    best_valid_loss, epoch, global_step,
                                )

                        best_str = (
                            f'{best_valid_loss:.6f}'
                            if best_valid_loss is not None else 'nan'
                        )
                        warmup_tag = '' if track_stop else ' [warmup]'
                        patience_tag = (
                            f' patience={patience_counter}/{self.patience}'
                            if self.early_stop and track_stop else ''
                        )
                        self._log(
                            f'epoch={epoch} step={global_step} '
                            f'loss={loss.item():.6f} '
                            f'valid_loss={valid_loss:.6f} '
                            f'lr={lr:.6g} best_valid_loss={best_str}'
                            f'{patience_tag} '
                            f'train_time={train_duration:.2f}s'
                            f'{warmup_tag}'
                        )
                        self._record_step(
                            epoch=epoch,
                            step=global_step,
                            loss=float(loss.item()),
                            valid_loss=float(valid_loss),
                            lr=float(lr),
                            best_valid_loss=(
                                float(best_valid_loss)
                                if best_valid_loss is not None
                                else float('nan')
                            ),
                            patience=patience_counter,
                            is_best=is_best,
                            warmup=not track_stop,
                            train_time_sec=float(train_duration),
                            valid_time_sec=float(valid_batch_times[-1]),
                        )
                        self._tb_scalar(
                            'train/loss', float(loss.item()), global_step,
                        )
                        self._tb_scalar(
                            'valid/loss', float(valid_loss), global_step,
                        )
                        if best_valid_loss is not None:
                            self._tb_scalar(
                                'valid/best_loss',
                                float(best_valid_loss),
                                global_step,
                            )
                        self._tb_scalar('train/lr', float(lr), global_step)
                        self._tb_scalar(
                            'train/epoch', float(epoch), global_step,
                        )
                        self._tb_scalar(
                            'train/patience',
                            float(patience_counter),
                            global_step,
                        )
                        if self.tb_writer is not None:
                            self.tb_writer.flush()

                        if (
                            self.early_stop
                            and track_stop
                            and epoch >= self.early_stop_min_epochs
                            and patience_counter >= self.patience
                        ):
                            self._log(
                                f'Early stopping at epoch {epoch}, '
                                f'step {global_step} '
                                f'(patience={self.patience})'
                            )
                            early_stopped = True
                            stop = True
                            break

                        epoch_start = time.time()

                if stop:
                    break

                if self.lr_anneal_epochs and epoch % self.lr_anneal_epochs == 0:
                    lr *= self.lr_anneal_factor
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = lr
                    msg = f'Learning rate decreased to {lr}'
                    self._log(msg)
                    self._record_step(
                        event='lr_anneal',
                        epoch=epoch,
                        step=global_step,
                        lr=float(lr),
                    )
                    self._tb_scalar('train/lr', float(lr), global_step)
        finally:
            if self.tb_writer is not None:
                self.tb_writer.close()

        if best_state is not None:
            self.model.load_state_dict(best_state)

        metrics = {
            'best_valid_loss': float(
                best_valid_loss if best_valid_loss is not None else float('nan')
            ),
            'train_time': float(
                np.mean(train_batch_times) if train_batch_times else 0
            ),
            'valid_time': float(
                np.mean(valid_batch_times) if valid_batch_times else 0
            ),
            'stopped_epoch': epoch,
            'stopped_step': global_step,
            'early_stopped': early_stopped,
            'best_checkpoint': self.best_checkpoint_path,
        }
        if self.tb_dir:
            metrics['tensorboard_dir'] = self.tb_dir
        metrics.update(self._save_history())
        return metrics
