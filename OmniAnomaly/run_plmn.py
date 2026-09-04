#!/usr/bin/env python3
"""Train / score OmniAnomaly on a paperib PLMN and export UI predictions.

Example:
    cd OmniAnomaly
    ../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 10

Outputs:
    data/paperib/{PLMN}_*.pkl          — OmniAnomaly train/test arrays
    model/{PLMN}/{run}/                — checkpoints
    result/{PLMN}/{run}/               — scores + metrics
    ../data/predictions/{PLMN}_omnianomaly.json  — labeling UI overlay
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
    is_rate_metric,
    load_labels,
    load_plmn,
    metric_columns,
)


PRED_DIR = os.path.join(PAPERIB_ROOT, "data", "predictions")


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
    return [c for c in metric_columns(df) if not is_rate_metric(c)]


def _human_anomaly_mask(df: pd.DataFrame, plmn: str) -> np.ndarray:
    """True where a human range/point anomaly covers the sample time."""
    doc = load_labels(plmn)
    mask = np.zeros(len(df), dtype=bool)
    times = pd.to_datetime(df["time"], utc=True)
    for item in doc.get("labels") or []:
        if (item.get("tag") or "anomaly") != "anomaly":
            continue
        start = pd.to_datetime(item["start"], utc=True)
        end = pd.to_datetime(item.get("end") or item["start"], utc=True)
        mask |= (times >= start) & (times <= end)
    return mask


def prepare_arrays(
    plmn: str,
    *,
    train_ratio: float = 0.7,
    drop_labeled_from_train: bool = True,
) -> dict:
    df = load_plmn(plmn)
    cols = _feature_columns(df)
    if not cols:
        raise SystemExit(f"no feature columns for {plmn}")
    values = df[cols].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    times = pd.to_datetime(df["time"], utc=True).to_numpy()

    n = len(df)
    split = max(int(n * train_ratio), 1)
    split = min(split, n - 1)

    train_raw = values[:split].copy()
    test_raw = values[split:].copy()
    y_test = np.zeros(len(test_raw), dtype=bool)
    human = _human_anomaly_mask(df, plmn)
    y_test = human[split:]

    if drop_labeled_from_train:
        keep = ~human[:split]
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
                f"{keep.sum()}/{split}",
                flush=True,
            )

    scaler = MinMaxScaler()
    train = scaler.fit_transform(train_raw).astype(np.float32)
    test = scaler.transform(test_raw).astype(np.float32)
    full = scaler.transform(values).astype(np.float32)

    data_dir = os.path.join(ROOT, "data", "paperib")
    os.makedirs(data_dir, exist_ok=True)
    meta = {
        "plmn": plmn,
        "x_dim": int(train.shape[1]),
        "feature_columns": cols,
        "train_ratio": train_ratio,
        "n_total": n,
        "n_train_raw_split": split,
        "n_train": int(train.shape[0]),
        "n_test": int(test.shape[0]),
        "drop_labeled_from_train": drop_labeled_from_train,
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "split_time": str(times[split]),
    }
    with open(os.path.join(data_dir, f"{plmn}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(data_dir, f"{plmn}_train.pkl"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(data_dir, f"{plmn}_test.pkl"), "wb") as f:
        pickle.dump(test, f)
    with open(os.path.join(data_dir, f"{plmn}_test_label.pkl"), "wb") as f:
        pickle.dump(y_test.astype(np.int8), f)
    with open(os.path.join(data_dir, f"{plmn}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(data_dir, f"{plmn}_full.pkl"), "wb") as f:
        pickle.dump(full, f)
    with open(os.path.join(data_dir, f"{plmn}_times.pkl"), "wb") as f:
        pickle.dump(times, f)

    print(json.dumps(meta, indent=2), flush=True)
    return {
        "train": train,
        "test": test,
        "y_test": y_test,
        "full": full,
        "times": times,
        "meta": meta,
        "cols": cols,
        "split": split,
        "human": human,
    }


def _pot_threshold(train_score: np.ndarray, score: np.ndarray, *, q: float, level: float) -> float:
    s = SPOT(q)
    s.fit(train_score, score)
    s.initialize(level=level, min_extrema=True)
    return -float(s.extreme_quantile)


def _preds_to_label_items(
    times: np.ndarray,
    pred: np.ndarray,
    scores: np.ndarray,
    *,
    window_length: int,
) -> list[dict]:
    """Map score-aligned boolean preds → UI label items (UTC ISO)."""
    # score[i] aligns with times[i + window_length - 1]
    offset = int(window_length) - 1
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
        score_slice = scores[max(0, i - offset) : max(0, j - offset)]
        score_min = float(np.min(score_slice)) if len(score_slice) else None
        kind = "point" if j - i == 1 else "range"
        items.append(
            {
                "id": f"oa_{i:06d}",
                "kind": kind,
                "tag": "model",
                "start": start_ts.isoformat(),
                "end": end_ts.isoformat(),
                "metrics": ["ALL"],
                "score": score_min,
                "source": "omnianomaly",
            }
        )
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
) -> str:
    pred = np.asarray(scores).reshape(-1) < float(threshold)
    labels = _preds_to_label_items(times, pred, scores, window_length=window_length)
    payload = {
        "plmn": plmn,
        "source": "omnianomaly",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "threshold_method": "pot",
        "window_length": int(window_length),
        "n_scores": int(len(scores)),
        "n_pred_points": int(pred.sum()),
        "n_pred_segments": len(labels),
        "feature_columns": meta.get("feature_columns"),
        "metrics": metrics or {},
        "labels": labels,
    }
    os.makedirs(PRED_DIR, exist_ok=True)
    out_path = os.path.join(PRED_DIR, f"{plmn}_omnianomaly.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
    print(f"wrote UI predictions → {out_path} ({len(labels)} segments)", flush=True)
    return out_path


def run(plmn: str, args: argparse.Namespace) -> None:
    prepared = prepare_arrays(
        plmn,
        train_ratio=args.train_ratio,
        drop_labeled_from_train=not args.keep_labeled_in_train,
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

    resolve_output_dirs(config, run_name=config.run_name)
    os.makedirs(config.result_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"device={device}  x_dim={config.x_dim}  epochs={config.max_epoch}", flush=True)

    model = OmniAnomaly(config).to(device)
    metrics: dict = {}

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
        # Try default save dir
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
    train_score, _, _ = predictor.get_score(prepared["train"])
    test_score, _, pred_time = predictor.get_score(prepared["test"])
    full_score, _, _ = predictor.get_score(prepared["full"])
    metrics["pred_total_time"] = pred_time

    with open(os.path.join(config.result_dir, config.train_score_filename), "wb") as f:
        pickle.dump(train_score, f)
    with open(os.path.join(config.result_dir, config.test_score_filename), "wb") as f:
        pickle.dump(test_score, f)
    with open(os.path.join(config.result_dir, "full_score.pkl"), "wb") as f:
        pickle.dump(full_score, f)

    y_test = prepared["y_test"]
    # Align label length to scores (window shrink).
    wl = config.window_length
    y_aligned = y_test[wl - 1 :][: len(test_score)]
    if len(y_aligned) != len(test_score):
        y_aligned = np.zeros(len(test_score), dtype=bool)

    try:
        threshold = _pot_threshold(
            train_score.reshape(-1),
            full_score.reshape(-1),
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

    if y_aligned.any():
        try:
            pot_metrics = pot_eval(
                train_score.reshape(-1),
                test_score.reshape(-1),
                y_aligned,
                q=config.pot_q,
                level=float(config.level),
            )
            metrics.update(pot_metrics)
            threshold = float(pot_metrics["pot-threshold"])
        except Exception as exc:
            print(f"POT eval with labels failed: {exc}", flush=True)
        if args.eval_bf:
            try:
                bf_m, bf_th = bf_search(
                    test_score.reshape(-1),
                    y_aligned,
                    start=config.bf_search_min,
                    end=config.bf_search_max,
                    step_num=int(
                        (config.bf_search_max - config.bf_search_min)
                        / config.bf_search_step_size
                    ),
                    verbose=False,
                )
                metrics["bf-f1"] = float(bf_m[0])
                metrics["bf-precision"] = float(bf_m[1])
                metrics["bf-recall"] = float(bf_m[2])
                metrics["bf-threshold"] = float(bf_th)
            except Exception as exc:
                print(f"best-F1 search failed: {exc}", flush=True)

    with open(os.path.join(config.result_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2)
    with open(os.path.join(config.result_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(config.to_dict()), f, indent=2)

    export_ui_predictions(
        plmn,
        times=prepared["times"],
        scores=full_score.reshape(-1),
        threshold=float(threshold),
        window_length=wl,
        meta=prepared["meta"],
        metrics={k: metrics[k] for k in metrics if "curve" not in str(k)},
    )
    print("done.", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument("--max_epoch", type=int, default=10)
    p.add_argument("--window_length", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--run_name", default="paperib")
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
        help="Also run best-F1 threshold search on the test split (slow)",
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
