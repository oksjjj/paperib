# -*- coding: utf-8 -*-
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from omni_anomaly.spot import SPOT


def _prepare_rank_inputs(score, label):
    y_true = np.asarray(label).reshape(-1).astype(bool)
    score = np.asarray(score, dtype=float)
    if score.ndim > 1:
        score = score.sum(axis=-1)
    score = score.reshape(-1)
    if len(y_true) != len(score):
        raise ValueError('score and label must have the same length')
    return score, y_true


def _anomaly_segment_bounds(y_true):
    """Return (starts, ends) half-open index ranges for contiguous anomaly segments."""
    actual = np.asarray(y_true, dtype=bool).reshape(-1)
    padded = np.concatenate([[False], actual, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return starts, ends


def _point_adjusted_scores(score, starts, ends):
    """
    Convert raw scores to point-adjustment-equivalent scores.

    Under point adjustment (PA), an anomaly segment is detected whenever *any*
    point inside it crosses the threshold. Since a lower score means "more
    anomalous", that is equivalent to giving every point in the segment the
    segment's minimum score (its most anomalous point). Normal points keep
    their own score. Feeding these adjusted scores to standard sklearn ranking
    functions reproduces the PA confusion matrix at every threshold exactly.
    """
    adj = np.asarray(score, dtype=float).copy()
    for a, b in zip(starts, ends):
        adj[a:b] = adj[a:b].min()
    return adj


def _calc_pa_curves(score, y_true):
    """
    ROC / PR curves and AUC scores with point adjustment, via scikit-learn.

    ``score`` / ``y_true`` must already be prepared by ``_prepare_rank_inputs``.
    Lower score = more anomalous, so sklearn is fed ``-adjusted_score``.

    Returns:
        dict with auroc, auprc, and sklearn curve arrays.
    """
    starts, ends = _anomaly_segment_bounds(y_true)
    y_score = -_point_adjusted_scores(score, starts, ends)

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    return {
        'auroc': auroc,
        'auprc': auprc,
        'fpr': fpr,
        'tpr': tpr,
        'recall': recall,
        'precision': precision,
        'prevalence': float(y_true.mean()),
    }


def save_roc_pr_curves(curve, save_dir, prefix='roc_pr', dataset=None):
    """
    Save ROC and PR curve images (point-adjusted).

    Writes:
      ``{save_dir}/{prefix}_roc.png``
      ``{save_dir}/{prefix}_pr.png``
      ``{save_dir}/{prefix}_combined.png``
    """
    os.makedirs(save_dir, exist_ok=True)
    title_ds = f' ({dataset})' if dataset else ''
    auroc = curve['auroc']
    auprc = curve['auprc']
    prevalence = curve.get('prevalence', 0.0)

    paths = {}

    # ROC (sklearn RocCurveDisplay)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay(
        fpr=curve['fpr'], tpr=curve['tpr'], roc_auc=auroc,
    ).plot(ax=ax, name='OmniAnomaly (PA)', plot_chance_level=True,
           curve_kwargs={'color': '#1f77b4', 'lw': 2})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f'ROC curve — point adjustment{title_ds}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    roc_path = os.path.join(save_dir, f'{prefix}_roc.png')
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    paths['roc_curve'] = roc_path

    # PR (sklearn PrecisionRecallDisplay)
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay(
        precision=curve['precision'], recall=curve['recall'],
        average_precision=auprc,
    ).plot(ax=ax, name='OmniAnomaly (PA)',
           curve_kwargs={'color': '#d62728', 'lw': 2})
    ax.axhline(prevalence, color='k', ls='--', lw=1,
               label=f'prevalence={prevalence:.4f}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f'PR curve — point adjustment{title_ds}')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    pr_path = os.path.join(save_dir, f'{prefix}_pr.png')
    fig.tight_layout()
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    paths['pr_curve'] = pr_path

    # Combined
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay(
        fpr=curve['fpr'], tpr=curve['tpr'], roc_auc=auroc,
    ).plot(ax=axes[0], name='PA', plot_chance_level=True,
           curve_kwargs={'color': '#1f77b4', 'lw': 2})
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title(f'ROC — PA{title_ds}')
    axes[0].grid(True, alpha=0.3)

    PrecisionRecallDisplay(
        precision=curve['precision'], recall=curve['recall'],
        average_precision=auprc,
    ).plot(ax=axes[1], name='PA',
           curve_kwargs={'color': '#d62728', 'lw': 2})
    axes[1].axhline(prevalence, color='k', ls='--', lw=1,
                    label=f'prevalence={prevalence:.4f}')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title(f'PR — PA{title_ds}')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)

    combined_path = os.path.join(save_dir, f'{prefix}_combined.png')
    fig.tight_layout()
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    paths['roc_pr_combined'] = combined_path

    print(f'ROC curve saved to {roc_path}')
    print(f'PR curve saved to {pr_path}')
    print(f'Combined curves saved to {combined_path}')
    return paths


def calc_rank_metrics(score, label, save_dir=None, dataset=None,
                      prefix='roc_pr'):
    """
    AUROC / AUPRC with point adjustment, computed via scikit-learn.

    Point adjustment is applied by mapping each anomaly segment to its most
    anomalous (minimum) score, which makes standard sklearn ranking functions
    (``roc_auc_score``, ``average_precision_score``, ``roc_curve``,
    ``precision_recall_curve``) reproduce the PA confusion matrix exactly.
    AUPRC is sklearn's Average Precision. If ``save_dir`` is set, ROC/PR curve
    images are written there.
    """
    score, y_true = _prepare_rank_inputs(score, label)
    if len(np.unique(y_true)) < 2:
        nan = float('nan')
        return {'auroc': nan, 'auprc': nan, 'point_adjustment': True}

    curve = _calc_pa_curves(score, y_true)
    out = {
        'auroc': float(curve['auroc']),
        'auprc': float(curve['auprc']),
        'point_adjustment': True,
    }
    if save_dir is not None:
        paths = save_roc_pr_curves(
            curve, save_dir, prefix=prefix, dataset=dataset,
        )
        out.update(paths)
    return out


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).


    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def pot_eval(init_score, score, label, q=1e-4, level=0.02):
    """
    Run POT method on given score.

    Args:
        init_score (np.ndarray): Anomaly scores of the train set (for init).
        score (np.ndarray): Anomaly scores of the test set.
        label: Ground-truth labels for the test set.
        q (float): Detection level / risk. Paper uses ``1e-4``.
        level (float): Low quantile for the initial threshold ``t``
            (SMAP 0.07, MSL 0.01, SMD subset-specific).

    Notes:
        Only ``SPOT.initialize()`` is used to obtain the GPD extreme quantile.
        ``SPOT.run()`` is intentionally skipped: with ``dynamic=False`` it
        overwrites that quantile, and streaming alarms are unused because
        predictions are made with a fixed threshold + point adjustment.
    """
    s = SPOT(q)
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=True)

    # Negated-score SPOT → original score threshold
    pot_th = -float(s.extreme_quantile)
    print('POT threshold (GPD extreme quantile):', pot_th,
          '(init_threshold on negated scores:', float(s.init_threshold), ')')

    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    print('POT result: ', p_t, pot_th, p_latency)
    return {
        'pot-f1': p_t[0],
        'pot-precision': p_t[1],
        'pot-recall': p_t[2],
        'pot-TP': p_t[3],
        'pot-TN': p_t[4],
        'pot-FP': p_t[5],
        'pot-FN': p_t[6],
        'pot-threshold': pot_th,
        'pot-latency': p_latency,
        'pot-q': q,
        'pot-level': level,
        'point_adjustment': True,
    }
