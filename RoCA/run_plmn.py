#!/usr/bin/env python3
"""Train official-style RoCA on one PLMN; export valid predictions for labeling UI.

Uses upstream network + loss (ruiking04/RoCA, MIT) without Merlion.
Same paper metrics / UI JSON schema as RiAD for backbone comparison.

Example:
  ../.venv/bin/python run_plmn.py --plmn P0480 --epochs 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import RoCAConfig
from dataset import RoCAWindowDataset, block_to_nct
from export_predictions import write_roca_prediction
from metrics import evaluate_valid
from official.loss import center_c, roca_loss
from official.network import base_Model

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODEL_ROOT = ROOT / "model"


def _load_riad_prepare():
    path = REPO / "RiAD" / "data.py"
    spec = importlib.util.spec_from_file_location("riad_data_mod", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.prepare_plmn


prepare_plmn = _load_riad_prepare()


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_percentile_threshold(train_scores: np.ndarray, anomaly_ratio: float) -> float:
    finite = train_scores[np.isfinite(train_scores)]
    if finite.size == 0:
        raise RuntimeError("no finite train scores for threshold")
    q = max(0.0, min(100.0, 100.0 - float(anomaly_ratio)))
    return float(np.percentile(finite, q))


@torch.no_grad()
def score_block(
    model: base_Model,
    center: torch.Tensor,
    block: np.ndarray,
    *,
    cfg: RoCAConfig,
    device: torch.device,
    time_step: int | None = None,
) -> np.ndarray:
    """Point scores: RoCA ``loss_n`` at window ends (higher = more anomalous)."""
    model.eval()
    step = int(time_step if time_step is not None else cfg.time_step)
    t = len(block)
    scores = np.full(t, np.nan, dtype=np.float64)
    if t < cfg.window_size:
        return scores
    wins, ends = block_to_nct(block, window=cfg.window_size, time_step=step)
    if wins.shape[0] == 0:
        return scores
    loader = DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(wins)),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    outs: list[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.float().to(device)
        feat, feat_dec = model(xb)
        _, _, loss_n, _ = roca_loss(feat, feat_dec, center, epoch=cfg.warmup + 1, config=cfg, device=device)
        outs.append(loss_n.detach().cpu().numpy())
    err = np.concatenate(outs, axis=0)
    for e, s in zip(ends, err):
        if 0 <= int(e) < t:
            scores[int(e)] = float(s)
    return scores


def train_and_eval(
    *,
    plmn: str,
    epochs: int | None = None,
    seed: int = 2,
    anomaly_ratio: float | None = None,
    window: int | None = None,
    time_step: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    nu: float | None = None,
    export_ui: bool = True,
    save_checkpoint: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    prep = prepare_plmn(plmn, window=window or 32)
    n_feat = len(prep["cols"])
    cfg = RoCAConfig(input_channels=n_feat)
    if epochs is not None:
        cfg.num_epoch = int(epochs)
    if anomaly_ratio is not None:
        cfg.anomaly_ratio = float(anomaly_ratio)
    if window is not None:
        cfg.window_size = int(window)
    if time_step is not None:
        cfg.time_step = int(time_step)
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    if lr is not None:
        cfg.lr = float(lr)
    if nu is not None:
        cfg.nu = float(nu)

    device = _device()
    if not quiet:
        print(
            f"device={device} plmn={plmn} window={cfg.window_size} "
            f"step={cfg.time_step} epochs={cfg.num_epoch} ch={n_feat} "
            f"ar={cfg.anomaly_ratio} nu={cfg.nu}",
            flush=True,
        )

    train_nct, _ = block_to_nct(
        prep["train"], window=cfg.window_size, time_step=cfg.time_step
    )
    train_y = np.zeros(len(train_nct), dtype=np.int64)
    train_ds = RoCAWindowDataset(train_nct, train_y, cfg)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=cfg.drop_last,
        num_workers=0,
    )
    if not quiet:
        print(f"train_windows={len(train_ds)} valid_len={len(prep['valid'])}", flush=True)

    model = base_Model(cfg, device).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")
    center = torch.zeros(cfg.project_channels, device=device)
    center = center_c(train_dl, model, device, center, cfg, eps=cfg.center_eps)

    history: list[dict[str, float]] = []
    t0 = time.time()
    for epoch in range(1, cfg.num_epoch + 1):
        model.train()
        losses: list[float] = []
        for data, _target, aug1, aug2 in train_dl:
            data = data.float().to(device)
            aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
            all_data = torch.cat((data, aug1, aug2), dim=0)
            opt.zero_grad()
            feature1, feature_dec1 = model(all_data)
            loss_oc, loss_sigam, _loss_n, _loss_a = roca_loss(
                feature1, feature_dec1, center, epoch, cfg, device
            )
            loss = loss_oc + cfg.omega * loss_sigam
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if epoch < cfg.change_center_epoch or epoch == 1:
            center = center_c(train_dl, model, device, center, cfg, eps=cfg.center_eps)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        scheduler.step(mean_loss)
        history.append({"epoch": epoch, "loss": mean_loss})
        if not quiet:
            print(f"epoch {epoch}/{cfg.num_epoch} loss={mean_loss:.4f}", flush=True)

    # Dense scoring (stride 1) for VUS-comparable point series.
    train_scores = score_block(
        model, center, prep["train"], cfg=cfg, device=device, time_step=1
    )
    valid_scores = score_block(
        model, center, prep["valid"], cfg=cfg, device=device, time_step=1
    )
    thr = train_percentile_threshold(train_scores, cfg.anomaly_ratio)
    y_valid = prep["y_valid"]
    eval_bundle = evaluate_valid(
        y_valid, valid_scores, thr, vus_window=None, sample_minutes=5.0
    )

    out: dict[str, Any] = {
        "plmn": plmn,
        "threshold": thr,
        "metrics": {
            "vus_pr": eval_bundle["primary"]["vus_pr"],
            "event_f1": eval_bundle["event"].get("f1"),
            "pr_auc": eval_bundle["pr_auc"],
            "false_alarms_per_day": eval_bundle["operational"].get(
                "false_alarms_per_day"
            ),
            **{k: eval_bundle["point"].get(k) for k in ("precision", "recall", "f1")},
        },
        "config": {
            "window_size": cfg.window_size,
            "time_step": cfg.time_step,
            "epochs": cfg.num_epoch,
            "anomaly_ratio": cfg.anomaly_ratio,
            "nu": cfg.nu,
            "lr": cfg.lr,
            "seed": seed,
        },
        "history": history,
        "elapsed_sec": time.time() - t0,
    }

    run_dir = MODEL_ROOT / plmn / "paperib"
    run_dir.mkdir(parents=True, exist_ok=True)
    if save_checkpoint:
        ckpt = {
            "model": model.state_dict(),
            "center": center.detach().cpu(),
            "config": out["config"],
            "n_features": n_feat,
            "cols": prep["cols"],
        }
        torch.save(ckpt, run_dir / "roca.pt")
        (run_dir / "train_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    times_valid = prep["times"].iloc[prep["train_end"] : prep["valid_end"]]
    if export_ui:
        path = write_roca_prediction(
            plmn=plmn,
            times_valid=times_valid,
            scores_valid=valid_scores,
            y_valid=y_valid,
            threshold=thr,
            feature_columns=prep["cols"],
            window=cfg.window_size,
            split=prep["split"],
            metrics_extra={
                "model": "roca",
                "vus_window": eval_bundle.get("vus_window"),
                "score_distribution": eval_bundle.get("score_distribution"),
                "event_score_diagnostics": eval_bundle.get("event_score_diagnostics"),
                "upstream": "ruiking04/RoCA",
                "train_method": cfg.train_method,
                "nu": cfg.nu,
                "anomaly_ratio": cfg.anomaly_ratio,
                "seed": seed,
                "epochs": cfg.num_epoch,
                "lr": cfg.lr,
            },
        )
        out["pred_path"] = path
        if not quiet:
            print(f"exported {path}", flush=True)
            m = out["metrics"]
            print(
                f"VUS-PR={m['vus_pr']:.4f} (vus_window={eval_bundle.get('vus_window')}) "
                f"Event-F1={m['event_f1']} FA/day={m['false_alarms_per_day']}",
                flush=True,
            )
            sd = eval_bundle.get("score_distribution") or {}
            print(
                f"valid score>thr fraction={sd.get('fraction_above_threshold')} "
                f"q95={sd.get('q95')} max={sd.get('max')}",
                flush=True,
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--anomaly_ratio", type=float, default=1.0)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--time_step", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--no_export", action="store_true")
    args = p.parse_args()
    train_and_eval(
        plmn=args.plmn,
        epochs=args.epochs,
        seed=args.seed,
        anomaly_ratio=args.anomaly_ratio,
        window=args.window,
        time_step=args.time_step,
        batch_size=args.batch_size,
        lr=args.lr,
        nu=args.nu,
        export_ui=not args.no_export,
    )


if __name__ == "__main__":
    # Ensure package-local imports work when run as a script.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
