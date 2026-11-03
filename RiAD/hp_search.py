#!/usr/bin/env python3
"""Hyperparameter search for Individual RiAD (valid-only).

Default target: paper Operator 1 (``P0480``).

Primary sort key: **VUS-PR**, then Event-F1, then lower FA/day.

Examples:
  # small default grid (thr + a few train knobs)
  ../.venv/bin/python hp_search.py --plmn P0480

  # thr-only: train once, sweep anomaly_ratio (fast start)
  ../.venv/bin/python hp_search.py --mode thr --plmn P0480

  # random search
  ../.venv/bin/python hp_search.py --mode random --n_trials 20 --plmn P0480

  # export best trial to labeling UI JSON
  ../.venv/bin/python hp_search.py --mode grid --export_best
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from data import DEFAULT_WINDOW, prepare_plmn
from export_predictions import write_riad_prediction
from metrics import evaluate_riad_valid
from run_plmn import train_and_eval, train_percentile_threshold

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "model"


# --- search spaces ---------------------------------------------------------

DEFAULT_GRID: dict[str, list[Any]] = {
    # Stage A: thr (cheap if paired with fixed train; here full retrain per cell
    # unless --mode thr with --reuse_ckpt)
    "anomaly_ratio": [0.5, 1.0, 2.0, 5.0],
    "lr": [1e-3, 3e-4],
    "hidden": [64, 128],
    "epochs": [10],
    "inject_p": [0.0, 0.15],
    "window": [100],
    "latent": [32],
    "layers": [1],
    "batch_size": [256],
    "seed": [42],
}

# Thr-only grid (apply many thr on one trained model).
THR_GRID: dict[str, list[Any]] = {
    "anomaly_ratio": [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
}

RANDOM_SPACE: dict[str, list[Any]] = {
    "anomaly_ratio": [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
    "lr": [3e-4, 1e-3, 3e-3],
    "hidden": [32, 64, 128],
    "latent": [16, 32, 64],
    "epochs": [5, 10, 15],
    "inject_p": [0.0, 0.1, 0.15, 0.25],
    "window": [50, 100],
    "layers": [1],
    "batch_size": [256],
    "seed": [42, 43],
}


def _flatten_grid(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def _sample_random(space: dict[str, list[Any]], n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    keys = list(space.keys())
    out: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        params = {k: space[k][int(rng.integers(0, len(space[k])))] for k in keys}
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(params)
    return out


def _row_from_result(trial_id: int, result: dict[str, Any]) -> dict[str, Any]:
    m = result["metrics"]
    pt, ev, op = m["point"], m["event"], m["operational"]
    params = result["params"]
    return {
        "trial_id": trial_id,
        **params,
        "threshold": result["threshold"],
        "train_time": result["train_time"],
        "vus_pr": m["primary"]["vus_pr"],
        "event_f1": ev["f1"],
        "event_precision": ev["precision"],
        "event_recall": ev["recall"],
        "pr_auc": m["pr_auc"],
        "point_precision": pt["precision"],
        "point_recall": pt["recall"],
        "point_f1": pt["f1"],
        "point_tp": pt["tp"],
        "point_fp": pt["fp"],
        "point_fn": pt["fn"],
        "false_alarms_per_day": op["false_alarms_per_day"],
        "mean_detection_delay_min": op["mean_detection_delay_min"],
        "checkpoint": result.get("checkpoint"),
        "pred_path": result.get("pred_path"),
    }


def _sort_key(row: dict[str, Any]) -> tuple:
    vus = row.get("vus_pr")
    vus_v = float(vus) if isinstance(vus, (int, float)) and vus == vus else -1.0
    ef1 = float(row.get("event_f1") or 0.0)
    fa = float(row.get("false_alarms_per_day") or 1e9)
    return (-vus_v, -ef1, fa)


def _write_table(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    json_path = out_dir / "results.json"
    if not rows:
        return
    keys = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    ranked = sorted(rows, key=_sort_key)
    (out_dir / "best.json").write_text(
        json.dumps(ranked[0], indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"best → {out_dir / 'best.json'}", flush=True)


def _print_leaderboard(rows: list[dict[str, Any]], top_k: int = 5) -> None:
    ranked = sorted(rows, key=_sort_key)[:top_k]
    print("\n=== leaderboard (VUS-PR ↓, Event-F1 ↓, FA/day ↑) ===", flush=True)
    for i, r in enumerate(ranked, 1):
        print(
            f"{i}. trial={r['trial_id']}  VUS-PR={r['vus_pr']}  "
            f"Event-F1={r['event_f1']:.3f}  FA/day={r['false_alarms_per_day']:.3f}  "
            f"P={r['point_precision']:.3f} R={r['point_recall']:.3f}  "
            f"ar={r['anomaly_ratio']} lr={r['lr']} h={r['hidden']} "
            f"ep={r['epochs']} inj={r['inject_p']} win={r['window']}",
            flush=True,
        )


def run_grid_or_random(
    *,
    plmn: str,
    mode: str,
    n_trials: int,
    seed: int,
    export_best: bool,
    max_trials: int | None,
) -> Path:
    rng = np.random.default_rng(seed)
    if mode == "grid":
        trials = _flatten_grid(DEFAULT_GRID)
    else:
        trials = _sample_random(RANDOM_SPACE, n_trials, rng)
    if max_trials is not None:
        trials = trials[: max(0, int(max_trials))]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / plmn / "hp_search" / f"{mode}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "space.json").write_text(
        json.dumps(
            {"mode": mode, "n": len(trials), "grid": DEFAULT_GRID if mode == "grid" else RANDOM_SPACE},
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    prep_cache: dict[int, Any] = {}
    t_all = time.time()
    for i, params in enumerate(trials):
        print(f"\n----- trial {i + 1}/{len(trials)} {params} -----", flush=True)
        win = int(params.get("window", DEFAULT_WINDOW))
        if win not in prep_cache:
            prep_cache[win] = prepare_plmn(plmn, window=win)
        result = train_and_eval(
            plmn=plmn,
            epochs=int(params["epochs"]),
            window=win,
            hidden=int(params["hidden"]),
            layers=int(params.get("layers", 1)),
            latent=int(params.get("latent", 32)),
            batch_size=int(params.get("batch_size", 256)),
            lr=float(params["lr"]),
            anomaly_ratio=float(params["anomaly_ratio"]),
            inject_p=float(params["inject_p"]),
            seed=int(params.get("seed", seed)),
            run_name=f"hp_search/{mode}_{stamp}/trial_{i:03d}",
            export_ui=False,
            save_checkpoint=True,
            prep=prep_cache[win],
            quiet=True,
        )
        # drop heavy objects before summarizing
        light = {k: v for k, v in result.items() if k not in ("prep", "model", "valid_scores", "y_valid", "times_valid", "history")}
        row = _row_from_result(i, light)
        rows.append(row)
        _write_table(rows, out_dir)
        print(
            f"trial {i}: VUS-PR={row['vus_pr']} Event-F1={row['event_f1']:.3f} "
            f"FA/day={row['false_alarms_per_day']:.3f}",
            flush=True,
        )

    _print_leaderboard(rows)
    print(f"total wall time {time.time() - t_all:.1f}s", flush=True)

    if export_best and rows:
        best = sorted(rows, key=_sort_key)[0]
        print(f"\nre-running best trial {best['trial_id']} with UI export…", flush=True)
        train_and_eval(
            plmn=plmn,
            epochs=int(best["epochs"]),
            window=int(best["window"]),
            hidden=int(best["hidden"]),
            layers=int(best.get("layers", 1)),
            latent=int(best.get("latent", 32)),
            batch_size=int(best.get("batch_size", 256)),
            lr=float(best["lr"]),
            anomaly_ratio=float(best["anomaly_ratio"]),
            inject_p=float(best["inject_p"]),
            seed=int(best.get("seed", seed)),
            run_name="paperib",
            export_ui=True,
            save_checkpoint=True,
            quiet=False,
        )
    return out_dir


def run_thr_only(
    *,
    plmn: str,
    epochs: int,
    window: int,
    hidden: int,
    layers: int,
    latent: int,
    batch_size: int,
    lr: float,
    inject_p: float,
    seed: int,
    export_best: bool,
) -> Path:
    """Train once, sweep anomaly_ratio thresholds on fixed scores."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / plmn / "hp_search" / f"thr_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("training base model once for thr sweep…", flush=True)
    base = train_and_eval(
        plmn=plmn,
        epochs=epochs,
        window=window,
        hidden=hidden,
        layers=layers,
        latent=latent,
        batch_size=batch_size,
        lr=lr,
        anomaly_ratio=1.0,  # placeholder; overwritten per thr
        inject_p=inject_p,
        seed=seed,
        run_name=f"hp_search/thr_{stamp}/base",
        export_ui=False,
        save_checkpoint=True,
        quiet=False,
    )
    # Need train scores for thr; recompute quickly.
    from run_plmn import score_block as _sb

    device = base["model"].parameters().__next__().device
    train_scores = _sb(
        base["model"],
        base["prep"]["train"],
        window=window,
        batch_size=batch_size,
        device=device,
    )
    valid_scores = base["valid_scores"]
    y_valid = base["y_valid"]
    times_valid = base["times_valid"]

    rows: list[dict[str, Any]] = []
    for i, ar in enumerate(THR_GRID["anomaly_ratio"]):
        thr = train_percentile_threshold(train_scores, float(ar))
        bundle = evaluate_riad_valid(
            y_valid, valid_scores, thr, window=window, sample_minutes=5.0
        )
        result = {
            "params": {
                "epochs": epochs,
                "window": window,
                "hidden": hidden,
                "layers": layers,
                "latent": latent,
                "batch_size": batch_size,
                "lr": lr,
                "anomaly_ratio": float(ar),
                "inject_p": inject_p,
                "seed": seed,
            },
            "threshold": thr,
            "train_time": base["train_time"],
            "metrics": bundle,
            "checkpoint": base.get("checkpoint"),
            "pred_path": None,
        }
        row = _row_from_result(i, result)
        rows.append(row)
        print(
            f"ar={ar}: thr={thr:.6g} VUS-PR={row['vus_pr']} "
            f"Event-F1={row['event_f1']:.3f} FA/day={row['false_alarms_per_day']:.3f} "
            f"P={row['point_precision']:.3f} R={row['point_recall']:.3f}",
            flush=True,
        )

    _write_table(rows, out_dir)
    _print_leaderboard(rows)

    if export_best and rows:
        best = sorted(rows, key=_sort_key)[0]
        thr = float(best["threshold"])
        path = write_riad_prediction(
            plmn=plmn,
            times_valid=times_valid,
            scores_valid=valid_scores,
            y_valid=y_valid,
            threshold=thr,
            feature_columns=base["prep"]["cols"],
            window=window,
            split=base["prep"]["split"],
            metrics_extra={
                "anomaly_ratio": best["anomaly_ratio"],
                "threshold": thr,
                "hp_search": str(out_dir.relative_to(ROOT)),
                "note": "thr-only sweep best",
            },
        )
        print(f"exported best thr UI pred → {path}", flush=True)
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument(
        "--mode",
        choices=("grid", "random", "thr"),
        default="grid",
        help="grid=full DEFAULT_GRID; random=RANDOM_SPACE; thr=train once then sweep thr",
    )
    p.add_argument("--n_trials", type=int, default=20, help="random mode only")
    p.add_argument("--max_trials", type=int, default=None, help="cap grid/random trials")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--export_best",
        action="store_true",
        help="retrain/export best trial to data/ib_data/predictions/{plmn}_riad.json",
    )
    # base train knobs for --mode thr
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--inject_p", type=float, default=0.15)
    args = p.parse_args()

    if args.mode == "thr":
        run_thr_only(
            plmn=args.plmn,
            epochs=args.epochs,
            window=args.window,
            hidden=args.hidden,
            layers=args.layers,
            latent=args.latent,
            batch_size=args.batch_size,
            lr=args.lr,
            inject_p=args.inject_p,
            seed=args.seed,
            export_best=args.export_best,
        )
    else:
        run_grid_or_random(
            plmn=args.plmn,
            mode=args.mode,
            n_trials=args.n_trials,
            seed=args.seed,
            export_best=args.export_best,
            max_trials=args.max_trials,
        )


if __name__ == "__main__":
    main()
