#!/usr/bin/env python3
"""Train / score OmniAnomaly on a paperib PLMN (train / valid / test).

Chronological split; human labels never tune on **test**.
By default only **valid** is scored for detection metrics + UI overlay.

Example (all metrics — run ``paperib``):
    cd OmniAnomaly
    ../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 10 --stable_train

Example (COMB subset — run ``comb``, raw counters + rates, MinMax):
    ../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 10 --stable_train \\
      --run_name comb

Example (COMB, window 200 — keeps default win=100 artifacts):
    ../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 10 --stable_train \\
      --run_name comb --window_length 200

Example (COMB per-attempt shares — run ``comb_share``):
    ../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 10 --stable_train \\
      --run_name comb_share

See ``COMB_RUNS.md`` for why these runs exist.

Outputs (``window_length`` default 100; non-default appends ``_w{N}``):
    ../data/ib_data/predictions/{PLMN}_omnianomaly.json              — paperib
    ../data/ib_data/predictions/{PLMN}_omnianomaly_comb.json         — comb
    ../data/ib_data/predictions/{PLMN}_omnianomaly_comb_w200.json    — comb, win=200
    ../data/ib_data/predictions/{PLMN}_omnianomaly_comb_share.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPERIB_ROOT = os.path.dirname(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
LABELING_ROOT = os.path.join(PAPERIB_ROOT, "labeling")
if LABELING_ROOT not in sys.path:
    sys.path.insert(0, LABELING_ROOT)

os.chdir(ROOT)

import torch  # noqa: E402

from omni_anomaly.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from omni_anomaly.device import get_device  # noqa: E402
from omni_anomaly.eval_methods import bf_search, pot_eval  # noqa: E402
from omni_anomaly.model import OmniAnomaly  # noqa: E402
from omni_anomaly.prediction import Predictor  # noqa: E402
from omni_anomaly.spot import SPOT  # noqa: E402
from omni_anomaly.training import Trainer  # noqa: E402
from omni_anomaly.utils import (  # noqa: E402
    default_pot_level,
    resolve_output_dirs,
)

from tool import (  # noqa: E402
    A_RATE_KEY,
    M971_COL,
    S_RATE_KEY,
    ensure_score_column,
    is_rate_metric,
    load_labels,
    load_plmn,
    load_predictions,
    metric_columns,
    metric_division_rate,
)


PRED_DIR = os.path.join(PAPERIB_ROOT, "data", "ib_data", "predictions")

# paperib (all-raw) training defaults: drop metrics that are almost never seen on
# train but appear sparsely on valid/test and are not operationally meaningful.
EXCLUDED_TRAIN_METRICS: frozenset[str] = frozenset({"M688"})

# --- COMB runs (see COMB_RUNS.md) ---
COMB_RUN_NAME = "comb"
COMB_SHARE_RUN_NAME = "comb_share"
COMB_RAW_COUNTERS: list[str] = [
    "M696",
    "M520",
    "M971",
    "M185",
    "M855",
    "M965",
    "M162",
    "M843",
    "M430",
    "M874",
    "M037",
    "M419",
    "M618",
    "M658",
]
# ``comb``: raw counters + SUCCESS/ACCEPT rates, MinMax (legacy).
COMB_LEGACY_COLUMNS: list[str] = list(COMB_RAW_COUNTERS) + [S_RATE_KEY, A_RATE_KEY]
# ``comb_share``: ATTEMPT + each counter ÷ ATTEMPT (14 dims), log1p+MinMax on ATTEMPT.
COMB_ENGINEERED_COLUMNS: list[str] = [M971_COL] + [
    c for c in COMB_RAW_COUNTERS if c != M971_COL
]
KNOWN_RUN_NAMES = frozenset({"paperib", COMB_RUN_NAME, COMB_SHARE_RUN_NAME})
DEFAULT_WINDOW_LENGTH = 100


def experiment_name_for_run(
    run_name: str, window_length: int = DEFAULT_WINDOW_LENGTH
) -> str:
    """Dir / artifact slug; non-default window gets ``_w{N}`` so win=100 stays intact."""
    base = run_name or "paperib"
    wl = int(window_length)
    if wl == DEFAULT_WINDOW_LENGTH:
        return base
    return f"{base}_w{wl}"


def _jsonable(obj):
    """Convert numpy / nested values to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


class PlmnConfig:
    """Minimal OmniAnomaly config for a paperib PLMN run."""

    dataset = "P0480"
    x_dim = 38
    use_connected_z_q = True
    use_connected_z_p = True
    include_prior_in_loss = True
    z_dim = 3
    rnn_cell = "GRU"
    rnn_num_hidden = 500
    window_length = 100
    dense_dim = 500
    posterior_flow_type = "nf"
    nf_layers = 20
    max_epoch = 10
    train_start = 0
    max_train_size = None
    batch_size = 50
    l2_reg = 0.0001
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4
    test_n_z = 1
    test_batch_size = 50
    test_start = 0
    max_test_size = None
    bf_search_min = -400.0
    bf_search_max = 400.0
    bf_search_step_size = 1.0
    valid_step_freq = 100
    gradient_clip_norm = 10.0
    grad_clip_mode = "per_tensor"
    early_stop = False
    early_stop_patience = 30
    early_stop_min_epochs = 3
    early_stop_warmup_steps = 300
    level = None
    pot_q = 1e-4
    save_z = False
    get_score_on_dim = False
    save_dir = "model"
    restore_dir = None
    result_dir = "result"
    train_score_filename = "train_score.pkl"
    valid_score_filename = "valid_score.pkl"
    test_score_filename = "test_score.pkl"
    experiment_name = None
    run_name = "paperib"
    log_dir = "log"
    device = None
    tensorboard = True

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k in dir(self)
            if not k.startswith("_") and not callable(getattr(self, k))
        }


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """All-metric (paperib) features: raw counters only, no rate overlays.

    Drops ``EXCLUDED_TRAIN_METRICS`` (operationally unimportant / mostly absent
    on train but sparse on valid·test — e.g. M688).
    """
    return [
        c
        for c in metric_columns(df)
        if not is_rate_metric(c) and c not in EXCLUDED_TRAIN_METRICS
    ]


def _comb_matrix(df: pd.DataFrame) -> np.ndarray:
    """M971 + each other selected counter divided by M971."""
    if M971_COL not in df.columns:
        raise SystemExit(f"comb requires {M971_COL} in dataframe")
    missing = [c for c in COMB_RAW_COUNTERS if c not in df.columns]
    if missing:
        raise SystemExit(f"comb raw counters missing: {missing}")
    attempt = df[M971_COL]
    parts = [attempt.astype(np.float64).to_numpy()]
    for col in COMB_ENGINEERED_COLUMNS[1:]:
        rate = metric_division_rate(df[col], attempt)
        parts.append(rate.astype(np.float64).to_numpy())
    mat = np.column_stack(parts)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def _scale_comb_splits(
    train_raw: np.ndarray,
    valid_raw: np.ndarray,
    test_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, dict]:
    """Scale comb matrix: log1p(ATTEMPT) then MinMax on train for all dims."""
    train = train_raw.astype(np.float64, copy=True)
    valid = valid_raw.astype(np.float64, copy=True)
    test = test_raw.astype(np.float64, copy=True)
    for block in (train, valid, test):
        block[:, 0] = np.log1p(np.maximum(block[:, 0], 0.0))
    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(train).astype(np.float32)
    valid_s = scaler.transform(valid).astype(np.float32)
    test_s = scaler.transform(test).astype(np.float32)
    scaling = {
        "method": "comb_per_attempt",
        "attempt_column": M971_COL,
        "attempt_transform": "log1p",
        "per_attempt": "counter_div_attempt",
        "post_transform": "minmax_train_fit",
        "note": (
            "col0=M971 (log1p then minmax); "
            "cols1+=counter/M971 then minmax (fit on train only)"
        ),
    }
    return train_s, valid_s, test_s, scaler, scaling


def features_for_run(run_name: str, df: pd.DataFrame) -> list[str] | None:
    """Resolve feature columns for a run.

    ``paperib`` → all raw metrics except ``EXCLUDED_TRAIN_METRICS``
    (``None`` means use ``_feature_columns``).
    ``comb`` → COMB_LEGACY_COLUMNS (raw + S_RATE/A_RATE).
    ``comb_share`` → M971 + counter÷M971 (COMB_ENGINEERED_COLUMNS).
    """
    if run_name not in KNOWN_RUN_NAMES:
        raise SystemExit(
            f"unknown --run_name {run_name!r}; "
            f"use one of: {', '.join(sorted(KNOWN_RUN_NAMES))}"
        )
    if run_name == "paperib":
        return None
    available = set(metric_columns(df))
    missing = [c for c in COMB_RAW_COUNTERS if c not in available]
    if missing:
        raise SystemExit(f"comb raw counters missing from dataframe columns: {missing}")
    if run_name == COMB_RUN_NAME:
        missing_rates = [c for c in (S_RATE_KEY, A_RATE_KEY) if c not in available]
        if missing_rates:
            raise SystemExit(
                f"comb requires rate columns {missing_rates} "
                "(load_plmn should add S_RATE/A_RATE)"
            )
        return list(COMB_LEGACY_COLUMNS)
    if run_name == COMB_SHARE_RUN_NAME:
        return list(COMB_ENGINEERED_COLUMNS)
    raise SystemExit(f"unhandled run_name {run_name!r}")


def pred_source_for_run(
    run_name: str, window_length: int = DEFAULT_WINDOW_LENGTH
) -> str:
    """UI prediction JSON stem: paperib → omnianomaly, else omnianomaly_{exp}.

    Non-default ``window_length`` appends ``_w{N}`` (e.g. ``omnianomaly_comb_w200``)
    so default win=100 files are not overwritten.
    """
    exp = experiment_name_for_run(run_name, window_length)
    if exp == "paperib":
        return "omnianomaly"
    if exp.startswith("paperib_"):
        return "omnianomaly" + exp[len("paperib") :]
    return f"omnianomaly_{exp}"


def _human_anomaly_mask(df: pd.DataFrame, plmn: str) -> np.ndarray:
    """True where a human range/point anomaly covers the sample time."""
    doc = load_labels(plmn)
    mask = np.zeros(len(df), dtype=bool)
    times = pd.to_datetime(df["time"], utc=True)
    for item in doc.get("labels") or []:
        start = pd.to_datetime(item["start"], utc=True)
        end = pd.to_datetime(item.get("end") or item["start"], utc=True)
        mask |= (times >= start) & (times <= end)
    return mask


def _split_indices(n: int, train_ratio: float, valid_ratio: float) -> tuple[int, int]:
    """Return (train_end, valid_end) exclusive indices for chronological splits."""
    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise SystemExit(
            f"need train_ratio>0, valid_ratio>0, train+valid<1; "
            f"got {train_ratio}, {valid_ratio}"
        )
    train_end = max(int(n * train_ratio), 1)
    valid_end = max(int(n * (train_ratio + valid_ratio)), train_end + 1)
    valid_end = min(valid_end, n - 1)
    if valid_end <= train_end or n - valid_end < 1:
        raise SystemExit(
            f"split too small for n={n}: train_end={train_end}, valid_end={valid_end}"
        )
    return train_end, valid_end


def prepare_arrays(
    plmn: str,
    *,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    drop_labeled_from_train: bool = True,
    feature_columns: list[str] | None = None,
    run_name: str = "paperib",
) -> dict:
    """Chronological train / valid / test. Scaler fit on train only."""
    df = load_plmn(plmn)
    available_all = metric_columns(df)
    available_raw = _feature_columns(df)
    if not available_raw:
        raise SystemExit(f"no feature columns for {plmn}")
    use_comb_share = run_name == COMB_SHARE_RUN_NAME
    use_comb_legacy = run_name == COMB_RUN_NAME
    if use_comb_share:
        cols = list(COMB_ENGINEERED_COLUMNS)
        values = _comb_matrix(df)
    elif use_comb_legacy:
        cols = list(COMB_LEGACY_COLUMNS)
        missing = [c for c in cols if c not in available_all]
        if missing:
            raise SystemExit(f"comb features not in {plmn}: {missing}")
        values = df[cols].to_numpy(dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        cols = list(feature_columns) if feature_columns else available_raw
        missing = [c for c in cols if c not in available_all]
        if missing:
            raise SystemExit(f"features not in {plmn}: {missing}")
        values = df[cols].to_numpy(dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    times = pd.to_datetime(df["time"], utc=True).to_numpy()

    n = len(df)
    train_end, valid_end = _split_indices(n, train_ratio, valid_ratio)
    human = _human_anomaly_mask(df, plmn)

    train_raw = values[:train_end].copy()
    valid_raw = values[train_end:valid_end].copy()
    test_raw = values[valid_end:].copy()
    y_valid = human[train_end:valid_end]
    y_test = human[valid_end:]

    if drop_labeled_from_train:
        keep = ~human[:train_end]
        if keep.sum() < max(200, PlmnConfig.window_length + 50):
            print(
                "warning: too few train rows after dropping labeled anomalies; "
                "keeping full train split",
                flush=True,
            )
        else:
            train_raw = train_raw[keep]
            print(
                f"train rows after dropping labeled anomalies: "
                f"{keep.sum()}/{train_end}",
                flush=True,
            )

    comb_scaling: dict | None = None
    if use_comb_share:
        train, valid, test, scaler, comb_scaling = _scale_comb_splits(
            train_raw, valid_raw, test_raw
        )
    else:
        scaler = MinMaxScaler()
        train = scaler.fit_transform(train_raw).astype(np.float32)
        valid = scaler.transform(valid_raw).astype(np.float32)
        test = scaler.transform(test_raw).astype(np.float32)
        if use_comb_legacy:
            comb_scaling = {
                "method": "minmax_train_fit",
                "note": "all COMB legacy columns (raw counters + S_RATE/A_RATE) MinMax on train",
            }

    if use_comb_share:
        feature_mode = "comb_share"
    elif use_comb_legacy:
        feature_mode = "comb"
    else:
        feature_mode = "all"

    # Per-run cache under shared ib_data/pickle (selected-feature runs do not overwrite).
    data_dir = os.path.join(
        PAPERIB_ROOT, "data", "ib_data", "pickle", run_name or "paperib"
    )
    os.makedirs(data_dir, exist_ok=True)
    meta = {
        "plmn": plmn,
        "run_name": run_name,
        "x_dim": int(train.shape[1]),
        "feature_columns": cols,
        "feature_mode": feature_mode,
        "excluded_train_metrics": sorted(EXCLUDED_TRAIN_METRICS)
        if feature_mode == "all"
        else [],
        "comb_scaling": comb_scaling,
        "comb_raw_counters": list(COMB_RAW_COUNTERS)
        if (use_comb_legacy or use_comb_share)
        else None,
        "n_features_available": len(available_raw),
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": round(1.0 - train_ratio - valid_ratio, 6),
        "n_total": n,
        "train_end": train_end,
        "valid_end": valid_end,
        "n_train": int(train.shape[0]),
        "n_valid": int(valid.shape[0]),
        "n_test": int(test.shape[0]),
        "n_human_train": int(human[:train_end].sum()),
        "n_human_valid": int(y_valid.sum()),
        "n_human_test": int(y_test.sum()),
        "drop_labeled_from_train": drop_labeled_from_train,
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "train_time_end": str(times[train_end - 1]),
        "valid_time_start": str(times[train_end]),
        "valid_time_end": str(times[valid_end - 1]),
        "test_time_start": str(times[valid_end]),
        "eval_split": "valid",
    }
    with open(os.path.join(data_dir, f"{plmn}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, f"{plmn}_train.pkl"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(data_dir, f"{plmn}_valid.pkl"), "wb") as f:
        pickle.dump(valid, f)
    with open(os.path.join(data_dir, f"{plmn}_test.pkl"), "wb") as f:
        pickle.dump(test, f)
    with open(os.path.join(data_dir, f"{plmn}_valid_label.pkl"), "wb") as f:
        pickle.dump(y_valid.astype(np.int8), f)
    with open(os.path.join(data_dir, f"{plmn}_test_label.pkl"), "wb") as f:
        pickle.dump(y_test.astype(np.int8), f)
    with open(os.path.join(data_dir, f"{plmn}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(data_dir, f"{plmn}_times.pkl"), "wb") as f:
        pickle.dump(times, f)

    print(json.dumps(meta, indent=2), flush=True)
    return {
        "train": train,
        "valid": valid,
        "test": test,
        "y_valid": y_valid,
        "y_test": y_test,
        "times": times,
        "times_valid": times[train_end:valid_end],
        "meta": meta,
        "cols": cols,
        "train_end": train_end,
        "valid_end": valid_end,
        "human": human,
        "run_name": run_name,
    }


def _pot_threshold(
    train_score: np.ndarray, score: np.ndarray, *, q: float, level: float
) -> float:
    s = SPOT(q)
    s.fit(train_score, score)
    s.initialize(level=level, min_extrema=True)
    return -float(s.extreme_quantile)


def _align_labels(y: np.ndarray, n_scores: int, window_length: int) -> np.ndarray:
    wl = int(window_length)
    y_aligned = y[wl - 1 :][:n_scores]
    if len(y_aligned) != n_scores:
        return np.zeros(n_scores, dtype=bool)
    return np.asarray(y_aligned, dtype=bool)


def _total_score(score: np.ndarray) -> np.ndarray:
    """Collapse per-dim scores to a scalar series (sum of log-probs)."""
    s = np.asarray(score)
    if s.ndim == 1:
        return s.reshape(-1)
    if s.ndim == 2:
        return s.sum(axis=-1)
    raise ValueError(f"unexpected score shape {s.shape}")


def _dim_baseline(train_score_dim: np.ndarray) -> np.ndarray:
    """Per-feature typical reconstruction log-prob on train (higher = more normal)."""
    return np.nanmedian(np.asarray(train_score_dim, dtype=np.float64), axis=0)


def _top_metric_attrs(
    score_dim_slice: np.ndarray,
    *,
    feature_columns: list[str],
    baseline: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """Rank features by how much worse than train median (contribution).

    ``score_dim_slice``: (T_seg, D) or (D,) reconstruction log-probs.
    contribution[d] = baseline[d] - mean_score[d]  (larger → more anomalous).
    """
    arr = np.asarray(score_dim_slice, dtype=np.float64)
    if arr.ndim == 1:
        mean_score = arr
    else:
        mean_score = np.nanmean(arr, axis=0)
    if len(mean_score) != len(feature_columns):
        return []
    contrib = baseline.reshape(-1) - mean_score.reshape(-1)
    order = np.argsort(-contrib)  # descending contribution
    out = []
    for rank, di in enumerate(order[: max(1, int(top_k))], start=1):
        out.append(
            {
                "rank": rank,
                "metric": str(feature_columns[int(di)]),
                "contribution": float(contrib[int(di)]),
                "log_prob": float(mean_score[int(di)]),
            }
        )
    return out


def _preds_to_label_items(
    times: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
    *,
    window_length: int,
    score_dim: np.ndarray | None = None,
    feature_columns: list[str] | None = None,
    baseline: np.ndarray | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Map score-aligned boolean preds → UI label items (UTC ISO)."""
    # score[i] aligns with times[i + window_length - 1]
    offset = int(window_length) - 1
    total = _total_score(scores)
    pred = np.asarray(pred).reshape(-1)
    aligned = np.zeros(len(times), dtype=bool)
    end = min(len(pred), len(times) - offset)
    if end <= 0:
        return []
    aligned[offset : offset + end] = pred[:end]

    items: list[dict] = []
    i = 0
    n = len(aligned)
    while i < n:
        if not aligned[i]:
            i += 1
            continue
        j = i + 1
        while j < n and aligned[j]:
            j += 1
        start_ts = pd.Timestamp(times[i]).tz_convert("UTC")
        end_ts = pd.Timestamp(times[j - 1]).tz_convert("UTC")
        score_slice = total[max(0, i - offset) : max(0, j - offset)]
        score_min = float(np.min(score_slice)) if len(score_slice) else None
        kind = "point" if j - i == 1 else "range"
        item = {
            "id": f"oa_{i:06d}",
            "kind": kind,
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "score": score_min,
            "source": "omnianomaly",
        }
        if (
            score_dim is not None
            and feature_columns is not None
            and baseline is not None
        ):
            dim_slice = score_dim[max(0, i - offset) : max(0, j - offset)]
            item["top_metrics"] = _top_metric_attrs(
                dim_slice,
                feature_columns=feature_columns,
                baseline=baseline,
                top_k=top_k,
            )
        items.append(item)
        i = j
    return items


def export_ui_predictions(
    plmn: str,
    *,
    times: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    window_length: int,
    meta: dict,
    metrics: dict | None = None,
    eval_split: str = "valid",
    score_dim: np.ndarray | None = None,
    feature_columns: list[str] | None = None,
    baseline: np.ndarray | None = None,
    top_k: int = 5,
    source: str = "omnianomaly",
) -> str:
    total = _total_score(scores)
    pred = total < float(threshold)
    labels = _preds_to_label_items(
        times,
        pred,
        total,
        window_length=window_length,
        score_dim=score_dim,
        feature_columns=feature_columns,
        baseline=baseline,
        top_k=top_k,
    )
    # score[i] aligns with times[i + window_length - 1]
    offset = int(window_length) - 1
    score_end = min(len(total), max(0, len(times) - offset))
    score_times = times[offset : offset + score_end]
    score_vals = np.asarray(total[:score_end], dtype=np.float64)
    payload = {
        "plmn": plmn,
        "source": source,
        "run_name": meta.get("run_name"),
        "eval_split": eval_split,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "threshold_method": "pot",
        "window_length": int(window_length),
        "n_scores": int(len(total)),
        "n_pred_points": int(pred.sum()),
        "n_pred_segments": len(labels),
        "feature_columns": meta.get("feature_columns"),
        "feature_mode": meta.get("feature_mode"),
        "comb_scaling": meta.get("comb_scaling"),
        "score_series": {
            "times": [
                pd.Timestamp(t).tz_convert("UTC").isoformat() for t in score_times
            ],
            "scores": [float(v) for v in score_vals],
            "threshold": float(threshold),
            "note": (
                "reconstruction log-prob sum (higher = more normal); "
                "anomaly when score < threshold"
            ),
        },
        "attribution": {
            "method": "train_median_minus_log_prob",
            "top_k": int(top_k),
            "note": (
                "contribution = train_median(log_prob) - segment_mean(log_prob); "
                "larger means that metric reconstructed worse than usual"
            ),
        },
        "split": {
            "train_time_end": meta.get("train_time_end"),
            "valid_time_start": meta.get("valid_time_start"),
            "valid_time_end": meta.get("valid_time_end"),
            "test_time_start": meta.get("test_time_start"),
        },
        "metrics": metrics or {},
        "labels": labels,
    }
    os.makedirs(PRED_DIR, exist_ok=True)
    out_path = os.path.join(PRED_DIR, f"{plmn}_{source}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
    print(
        f"wrote UI predictions ({eval_split} only) → {out_path} "
        f"({len(labels)} segments)",
        flush=True,
    )
    if labels and labels[0].get("top_metrics"):
        tops = ", ".join(
            f"{t['metric']}({t['contribution']:.2f})" for t in labels[0]["top_metrics"][:3]
        )
        print(f"  example attribution (first segment): {tops}", flush=True)
    # Warm labeling UI score column cache (aligned 1:1 with PLMN frame).
    try:
        preds = load_predictions(plmn, source=source)
        df = load_plmn(plmn)
        ensure_score_column(df, preds, source=source, plmn=plmn)
        print(f"warmed score cache → labeling/cache/{plmn}_score_{source}.pkl", flush=True)
    except Exception as exc:
        print(f"score cache warmup skipped: {exc}", flush=True)
    return out_path


def run(plmn: str, args: argparse.Namespace) -> None:
    if args.run_name not in KNOWN_RUN_NAMES:
        raise SystemExit(
            f"unknown --run_name {args.run_name!r}; "
            f"use one of: {', '.join(sorted(KNOWN_RUN_NAMES))}"
        )
    df_probe = load_plmn(plmn)
    selected = features_for_run(args.run_name, df_probe)
    if selected is not None:
        print(
            f"run={args.run_name} features ({len(selected)}): {', '.join(selected)}",
            flush=True,
        )
    else:
        print(f"run={args.run_name} features: all raw metrics", flush=True)

    prepared = prepare_arrays(
        plmn,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        drop_labeled_from_train=not args.keep_labeled_in_train,
        feature_columns=selected,
        run_name=args.run_name,
    )
    config = PlmnConfig()
    config.dataset = plmn
    config.x_dim = int(prepared["meta"]["x_dim"])
    config.max_epoch = int(args.max_epoch)
    config.window_length = int(args.window_length)
    config.batch_size = int(args.batch_size)
    config.run_name = args.run_name
    config.level = args.level if args.level is not None else default_pot_level(plmn)
    config.pot_q = float(args.pot_q)
    if args.stable_train:
        config.std_epsilon = 1e-3
        config.initial_lr = 5e-4
        config.grad_clip_mode = "global"
        config.gradient_clip_norm = 5.0
    if args.restore_dir:
        config.restore_dir = args.restore_dir
    config.tensorboard = not args.no_tensorboard

    resolve_output_dirs(
        config,
        run_name=experiment_name_for_run(config.run_name, config.window_length),
    )
    os.makedirs(config.result_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"device={device}  x_dim={config.x_dim}  epochs={config.max_epoch}", flush=True)

    model = OmniAnomaly(config).to(device)
    metrics: dict = {"eval_split": "valid", "test_held_out": True}

    if config.restore_dir and config.max_epoch <= 0:
        load_checkpoint(model, config.restore_dir, device=device)
        print(f"restored from {config.restore_dir}", flush=True)
    elif config.max_epoch > 0:
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
            log_dir=config.log_dir,
            dataset=config.dataset,
            checkpoint_dir=config.save_dir,
            config=config.to_dict(),
            tensorboard=config.tensorboard,
        )
        # Trainer's internal valid_portion is a holdout from the train array only
        # (early-stop). Chronological "valid" below is for detection eval / UI.
        train_metrics = trainer.fit(prepared["train"])
        metrics.update(train_metrics or {})
        save_checkpoint(
            model,
            config.to_dict(),
            config.save_dir,
            filename="model.pt",
        )
        print(f"saved model → {config.save_dir}", flush=True)
        tb_dir = metrics.get("tensorboard_dir") or getattr(trainer, "tb_dir", None)
        if tb_dir:
            log_root = os.path.join("log", plmn)
            print(
                f"TensorBoard: {tb_dir}\n"
                f"  view: ../.venv/bin/python view_tensorboard.py --plmn {plmn}\n"
                f"  or:   ../.venv/bin/tensorboard --logdir {log_root}",
                flush=True,
            )
    else:
        try:
            load_checkpoint(model, config.save_dir, device=device)
            print(f"restored from {config.save_dir}", flush=True)
        except FileNotFoundError as exc:
            raise SystemExit(
                "no model to restore; pass --max_epoch > 0 or --restore_dir"
            ) from exc

    predictor = Predictor(
        model,
        device,
        batch_size=config.test_batch_size,
        n_z=config.test_n_z,
        last_point_only=True,
    )
    # Per-dimension reconstruction log-probs for attribution; total = sum(dims).
    config.get_score_on_dim = True
    train_score_raw, _, _ = predictor.get_score(prepared["train"])
    valid_score_raw, _, pred_time = predictor.get_score(prepared["valid"])
    metrics["pred_total_time"] = pred_time
    train_score = _total_score(train_score_raw)
    valid_score = _total_score(valid_score_raw)
    train_score_dim = (
        np.asarray(train_score_raw)
        if np.asarray(train_score_raw).ndim == 2
        else None
    )
    valid_score_dim = (
        np.asarray(valid_score_raw)
        if np.asarray(valid_score_raw).ndim == 2
        else None
    )
    print(
        "scored train + valid (per-dim attribution on); test left untouched "
        "(hold out until hyperparameters are fixed)",
        flush=True,
    )

    with open(os.path.join(config.result_dir, config.train_score_filename), "wb") as f:
        pickle.dump(train_score, f)
    with open(os.path.join(config.result_dir, config.valid_score_filename), "wb") as f:
        pickle.dump(valid_score, f)
    if train_score_dim is not None:
        with open(os.path.join(config.result_dir, "train_score_dim.pkl"), "wb") as f:
            pickle.dump(train_score_dim, f)
    if valid_score_dim is not None:
        with open(os.path.join(config.result_dir, "valid_score_dim.pkl"), "wb") as f:
            pickle.dump(valid_score_dim, f)

    wl = config.window_length
    y_valid = _align_labels(prepared["y_valid"], len(valid_score), wl)

    try:
        threshold = _pot_threshold(
            train_score.reshape(-1),
            valid_score.reshape(-1),
            q=config.pot_q,
            level=float(config.level),
        )
        metrics["pot-threshold"] = threshold
        metrics["pot-level"] = float(config.level)
        metrics["pot-q"] = float(config.pot_q)
    except Exception as exc:
        print(f"POT failed ({exc}); falling back to train 0.5% quantile", flush=True)
        threshold = float(np.quantile(train_score.reshape(-1), 0.005))
        metrics["pot-threshold"] = threshold
        metrics["threshold_fallback"] = "train_q0.005"

    if y_valid.any():
        try:
            pot_metrics = pot_eval(
                train_score.reshape(-1),
                valid_score.reshape(-1),
                y_valid,
                q=config.pot_q,
                level=float(config.level),
            )
            metrics.update({f"valid-{k}": v for k, v in pot_metrics.items()})
            threshold = float(pot_metrics["pot-threshold"])
            metrics["pot-threshold"] = threshold
        except Exception as exc:
            print(f"POT eval on valid failed: {exc}", flush=True)
        if args.eval_bf:
            try:
                bf_m, bf_th = bf_search(
                    valid_score.reshape(-1),
                    y_valid,
                    start=config.bf_search_min,
                    end=config.bf_search_max,
                    step_num=int(
                        (config.bf_search_max - config.bf_search_min)
                        / config.bf_search_step_size
                    ),
                    verbose=False,
                )
                metrics["valid-bf-f1"] = float(bf_m[0])
                metrics["valid-bf-precision"] = float(bf_m[1])
                metrics["valid-bf-recall"] = float(bf_m[2])
                metrics["valid-bf-threshold"] = float(bf_th)
            except Exception as exc:
                print(f"best-F1 search on valid failed: {exc}", flush=True)
        elif "valid-bf-threshold" not in metrics and y_valid.any():
            # Fast oracle thr for the labeling UI (grid over observed score range).
            try:
                vs = valid_score.reshape(-1)
                lo, hi = float(np.min(vs)), float(np.max(vs))
                if np.isfinite(lo) and hi > lo:
                    bf_m, bf_th = bf_search(
                        vs,
                        y_valid,
                        start=lo,
                        end=hi,
                        step_num=200,
                        verbose=False,
                    )
                    metrics["valid-bf-f1"] = float(bf_m[0])
                    metrics["valid-bf-precision"] = float(bf_m[1])
                    metrics["valid-bf-recall"] = float(bf_m[2])
                    metrics["valid-bf-threshold"] = float(bf_th)
            except Exception as exc:
                print(f"fast best-F1 for UI failed: {exc}", flush=True)
    else:
        print("no human labels in valid split; skipping labeled POT/F1", flush=True)

    pred_valid = valid_score.reshape(-1) < float(threshold)
    metrics["valid-n-pred"] = int(pred_valid.sum())
    metrics["valid-n-human"] = int(y_valid.sum())
    print(
        f"valid: human={int(y_valid.sum())}  pred={int(pred_valid.sum())}  "
        f"threshold={threshold:.4f}",
        flush=True,
    )
    for key in (
        "valid-pot-f1",
        "valid-pot-precision",
        "valid-pot-recall",
        "valid-bf-f1",
    ):
        if key in metrics:
            print(f"  {key}={metrics[key]}", flush=True)

    with open(os.path.join(config.result_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2)
    with open(os.path.join(config.result_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(config.to_dict()), f, indent=2)

    baseline = (
        _dim_baseline(train_score_dim) if train_score_dim is not None else None
    )
    export_ui_predictions(
        plmn,
        times=prepared["times_valid"],
        scores=valid_score.reshape(-1),
        threshold=float(threshold),
        window_length=wl,
        meta=prepared["meta"],
        metrics={k: metrics[k] for k in metrics if "curve" not in str(k)},
        eval_split="valid",
        score_dim=valid_score_dim,
        feature_columns=list(prepared["cols"]),
        baseline=baseline,
        top_k=int(args.attr_top_k),
        source=pred_source_for_run(args.run_name, wl),
    )
    print("done. (test split held out)", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument("--max_epoch", type=int, default=10)
    p.add_argument("--window_length", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--valid_ratio", type=float, default=0.2)
    p.add_argument(
        "--run_name",
        default="paperib",
        choices=sorted(KNOWN_RUN_NAMES),
        help="paperib=all raw; comb=selected raw+rates; comb_share=M971+÷M971 (see COMB_RUNS.md)",
    )
    p.add_argument("--level", type=float, default=None)
    p.add_argument("--pot_q", type=float, default=1e-4)
    p.add_argument("--device", default=None)
    p.add_argument("--restore_dir", default=None)
    p.add_argument("--stable_train", action="store_true")
    p.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable TensorBoard logging during training (on by default)",
    )
    p.add_argument(
        "--eval_bf",
        action="store_true",
        help="Also run best-F1 threshold search on the valid split (slow)",
    )
    p.add_argument(
        "--attr_top_k",
        type=int,
        default=5,
        help="How many metrics to list per predicted segment (attribution)",
    )
    p.add_argument(
        "--keep_labeled_in_train",
        action="store_true",
        help="Do not drop human-labeled anomaly rows from the train split",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.plmn, args)
