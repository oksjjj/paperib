# -*- coding: utf-8 -*-
import csv
import json
import logging
import os
import sys
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime


class _TeeStream:
    """Duplicate writes to console and log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        if not data:
            return
        self.stream.write(data)
        self.stream.flush()
        normalized = data.replace('\r\n', '\n').replace('\r', '\n')
        self.log_file.write(normalized)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()


def _configure_logger(stream):
    """Single logger writing to the (teed) stdout stream."""
    root = logging.getLogger('omni_anomaly')
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.propagate = False

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    for name in ('omni_anomaly.train', 'omni_anomaly.eval'):
        child = logging.getLogger(name)
        child.handlers.clear()
        child.propagate = True

    return root


def _write_error(text, teed_stderr):
    """Write error text to teed stderr (console + log file)."""
    if not text.endswith('\n'):
        text += '\n'
    teed_stderr.write(text)


@contextmanager
def experiment_logging(log_dir, dataset, log_filename=None, mode='train'):
    """
    Capture stdout, stderr, exceptions, and warnings into a log file.

    Args:
        mode: ``'train'`` or ``'eval'`` — used in default filename
              ``{dataset}_{timestamp}_{mode}.log``.

    Yields:
        tuple: (log_path, logger)
    """
    os.makedirs(log_dir, exist_ok=True)
    if log_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode = mode if mode in ('train', 'eval') else 'train'
        log_filename = f'{dataset}_{timestamp}_{mode}.log'

    log_path = os.path.join(log_dir, log_filename)
    log_file = open(log_path, 'w', encoding='utf-8')

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    teed_stdout = _TeeStream(original_stdout, log_file)
    teed_stderr = _TeeStream(original_stderr, log_file)

    sys.stdout = teed_stdout
    sys.stderr = teed_stderr
    logger = _configure_logger(teed_stdout)

    original_excepthook = sys.excepthook
    original_showwarning = warnings.showwarning

    def _excepthook(exc_type, exc_value, exc_tb):
        tb_text = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        _write_error(tb_text, teed_stderr)
        try:
            logger.error('Uncaught exception: %s', exc_value)
        except Exception:
            pass
        original_excepthook(exc_type, exc_value, exc_tb)

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        text = warnings.formatwarning(message, category, filename, lineno, line)
        _write_error(text, teed_stderr)
        try:
            logger.warning('%s', text.rstrip())
        except Exception:
            pass

    sys.excepthook = _excepthook
    warnings.showwarning = _showwarning

    try:
        yield log_path, logger
    except BaseException as exc:
        tb_text = traceback.format_exc()
        _write_error(tb_text, teed_stderr)
        logger.error('Experiment failed: %s', exc)
        raise
    finally:
        sys.excepthook = original_excepthook
        warnings.showwarning = original_showwarning
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def setup_train_logging(log_dir, dataset, log_filename=None):
    """Backward-compatible helper (prefer experiment_logging context manager)."""
    os.makedirs(log_dir, exist_ok=True)
    if log_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'{dataset}_{timestamp}_train.log'
    log_path = os.path.join(log_dir, log_filename)
    logger = _configure_logger(sys.stdout)
    return log_path, logger


def get_logger(name='omni_anomaly'):
    return logging.getLogger(name)


class TrainHistory:
    """Collect and persist structured training metrics."""

    def __init__(self, log_dir, dataset):
        self.log_dir = log_dir
        self.dataset = dataset
        self.records = []
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_name = f'{dataset}_{timestamp}_train_history'

    def add(self, **kwargs):
        record = dict(kwargs)
        self.records.append(record)

    def save(self):
        if not self.records:
            return None, None

        json_path = os.path.join(self.log_dir, self.base_name + '.json')
        csv_path = os.path.join(self.log_dir, self.base_name + '.csv')

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, indent=2, default=str)

        fieldnames = sorted({k for r in self.records for k in r})
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        return json_path, csv_path
