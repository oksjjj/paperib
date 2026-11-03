#!/usr/bin/env python3
"""Train RiAD on one PLMN and export valid-only predictions for labeling UI.

Default PLMN = paper Operator 1 (``P0480``).

Example:
  ../.venv/bin/python run_plmn.py --plmn P0480 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import DEFAULT_WINDOW, prepare_plmn, windows_for_split
from export_predictions import write_riad_prediction
from metrics import evaluate_riad_valid
from model import RiADModel, riad_train_step

_REPO = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(__file__).resolve().parent / "model"


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def score_block(
    model: RiADModel,
    block: np.ndarray,
    *,
    window: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Point scores: reconstruction MSE at window end (higher = more anomalous)."""
    model.eval()
    t = len(block)
    scores = np.full(t, np.nan, dtype=np.float64)
    if t < window:
        return scores
    wins = windows_for_split(block, window)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(wins)),
        batch_size=batch_size,
        shuffle=False,
    )
    errs: list[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(device)
        out = model(xb)
        err = model.recon_error(xb, out["mu"]).detach().cpu().numpy()
        errs.append(err)
    err_all = np.concatenate(errs, axis=0)
    scores[window - 1 :] = err_all
    return scores


def train_percentile_threshold(
    train_scores: np.ndarray, anomaly_ratio: float
) -> float:
    """Higher-is-anomalous: thr = top ``anomaly_ratio``% cutoff on train scores."""
    finite = train_scores[np.isfinite(train_scores)]
    if finite.size == 0:
        raise RuntimeError("no finite train scores for threshold")
    q = max(0.0, min(100.0, 100.0 - float(anomaly_ratio)))
    return float(np.percentile(finite, q))


def train_and_eval(
    *,
    plmn: str,
    epochs: int,
    window: int,
    hidden: int,
    layers: int,
    latent: int,
    batch_size: int,
    lr: float,
    anomaly_ratio: float,
    inject_p: float,
    seed: int,
    run_name: str = "paperib",
    export_ui: bool = True,
    save_checkpoint: bool = True,
    prep: dict[str, Any] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Train RiAD, score valid, return metrics (+ optional UI export)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = _device()
    if not quiet:
        print(
            f"device={device} plmn={plmn} window={window} epochs={epochs} "
            f"hidden={hidden} lr={lr} ar={anomaly_ratio} inj={inject_p}",
            flush=True,
        )

    if prep is None or int(prep.get("window") or -1) != int(window):
        prep = prepare_plmn(plmn, window=window)
    train_wins = windows_for_split(prep["train"], window)
    if not quiet:
        print(
            f"features={len(prep['cols'])} train_windows={len(train_wins)} "
            f"valid_len={len(prep['valid'])}",
            flush=True,
        )

    model = RiADModel(
        len(prep["cols"]),
        hidden=hidden,
        layers=layers,
        latent=latent,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_wins)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_wins) > batch_size,
    )

    t0 = time.time()
    history: list[dict] = []
    model.train()
    for ep in range(1, epochs + 1):
        ep_stats = {"loss": 0.0, "nll": 0.0, "mem": 0.0, "inject_frac": 0.0, "n": 0}
        nn_clip = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            loss, stats = riad_train_step(model, xb, inject_p=inject_p)
            loss.backward()
            nn_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_stats["loss"] += stats["loss"]
            ep_stats["nll"] += stats["nll"]
            ep_stats["mem"] += stats["mem"]
            ep_stats["inject_frac"] += stats["inject_frac"]
            ep_stats["n"] += 1
        n = max(ep_stats["n"], 1)
        row = {
            "epoch": ep,
            "loss": ep_stats["loss"] / n,
            "nll": ep_stats["nll"] / n,
            "mem": ep_stats["mem"] / n,
            "inject_frac": ep_stats["inject_frac"] / n,
            "grad_norm": float(nn_clip) if torch.is_tensor(nn_clip) else float(nn_clip),
        }
        history.append(row)
        if not quiet:
            print(
                f"epoch {ep}/{epochs} loss={row['loss']:.4f} nll={row['nll']:.4f} "
                f"mem={row['mem']:.4f} inj={row['inject_frac']:.2f}",
                flush=True,
            )
    train_time = time.time() - t0

    train_scores = score_block(
        model, prep["train"], window=window, batch_size=batch_size, device=device
    )
    thr = train_percentile_threshold(train_scores, anomaly_ratio)
    valid_scores = score_block(
        model, prep["valid"], window=window, batch_size=batch_size, device=device
    )
    times_valid = prep["times"].iloc[prep["train_end"] : prep["valid_end"]]
    y_valid = prep["y_valid"]

    bundle = evaluate_riad_valid(
        y_valid, valid_scores, thr, window=window, sample_minutes=5.0
    )

    ckpt_path = None
    pred_path = None
    if save_checkpoint:
        ckpt_dir = MODEL_ROOT / plmn / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "riad.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "cols": prep["cols"],
                "window": window,
                "hidden": hidden,
                "layers": layers,
                "latent": latent,
                "threshold": thr,
                "anomaly_ratio": anomaly_ratio,
                "lr": lr,
                "epochs": epochs,
                "inject_p": inject_p,
                "seed": seed,
                "scaler_min": prep["scaler"].min_.tolist(),
                "scaler_scale": prep["scaler"].scale_.tolist(),
            },
            ckpt_path,
        )
        (ckpt_dir / "train_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    if export_ui:
        if ckpt_path is None:
            # still allow UI export without ckpt
            rel_ckpt = None
        else:
            rel_ckpt = str(ckpt_path.relative_to(_REPO))
        pred_path = write_riad_prediction(
            plmn=plmn,
            times_valid=times_valid,
            scores_valid=valid_scores,
            y_valid=y_valid,
            threshold=thr,
            feature_columns=prep["cols"],
            window=window,
            split=prep["split"],
            metrics_extra={
                "train_time": train_time,
                "anomaly_ratio": anomaly_ratio,
                "threshold": thr,
                "checkpoint": rel_ckpt,
                "device": str(device),
                "epochs": epochs,
                "hidden": hidden,
                "latent": latent,
                "lr": lr,
                "inject_p": inject_p,
                "seed": seed,
                "run_name": run_name,
            },
        )

    pt, ev, op = bundle["point"], bundle["event"], bundle["operational"]
    if not quiet:
        print(
            f"valid VUS-PR={bundle['primary']['vus_pr']}  "
            f"Event-F1={ev['f1']:.3f}  PR-AUC={bundle['pr_auc']}",
            flush=True,
        )
        print(
            f"valid point: P={pt['precision']:.3f} R={pt['recall']:.3f} "
            f"F1={pt['f1']:.3f} TP={pt['tp']} FP={pt['fp']} FN={pt['fn']}",
            flush=True,
        )
        print(
            f"FA/day={op['false_alarms_per_day']:.3f}  "
            f"mean_delay_min={op['mean_detection_delay_min']}",
            flush=True,
        )
        print(f"threshold={thr:.6g} (train top {anomaly_ratio}%)", flush=True)
        if pred_path:
            print(f"wrote {pred_path}", flush=True)
        if ckpt_path:
            print(f"checkpoint {ckpt_path}", flush=True)

    return {
        "plmn": plmn,
        "run_name": run_name,
        "params": {
            "epochs": epochs,
            "window": window,
            "hidden": hidden,
            "layers": layers,
            "latent": latent,
            "batch_size": batch_size,
            "lr": lr,
            "anomaly_ratio": anomaly_ratio,
            "inject_p": inject_p,
            "seed": seed,
        },
        "threshold": thr,
        "train_time": train_time,
        "metrics": bundle,
        "pred_path": pred_path,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "history": history,
        "prep": prep,
        "valid_scores": valid_scores,
        "y_valid": y_valid,
        "times_valid": times_valid,
        "model": model,
        "device": str(device),
    }


def run(**kwargs) -> dict[str, Any]:
    """Backward-compatible wrapper used by ``main``. """
    return train_and_eval(**kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480", help="paper Operator 1 by default")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--anomaly_ratio",
        type=float,
        default=1.0,
        help="train top-%% for primary thr (higher-is-anomalous; 1.0 → 99th pct)",
    )
    p.add_argument("--inject_p", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", default="paperib")
    args = p.parse_args()
    train_and_eval(
        plmn=args.plmn,
        epochs=args.epochs,
        window=args.window,
        hidden=args.hidden,
        layers=args.layers,
        latent=args.latent,
        batch_size=args.batch_size,
        lr=args.lr,
        anomaly_ratio=args.anomaly_ratio,
        inject_p=args.inject_p,
        seed=args.seed,
        run_name=args.run_name,
        export_ui=True,
        save_checkpoint=True,
    )


if __name__ == "__main__":
    main()
