#!/usr/bin/env python3
"""Diagnose MSRS false alarms using label-free dependency and persistence signals.

Purpose
-------
Before adding another branch to MSRS, test whether GT anomaly events and current
MSRS false-alarm events differ in:

1. cross-variable dependency violation,
2. temporal persistence,
3. number of strongly deviating variables.

All dependency models and robust references are fitted on the TRAIN split only.
Human labels are loaded only after train/valid diagnostic signals are computed.

Default input:
  data/ib_data/predictions/P0480_msrs_valid.json
  MSRS/model/P0480/paperib/msrs.pt

Outputs:
  data/ib_data/diagnostics/P0480_msrs_dependency_persistence.json
  data/ib_data/diagnostics/P0480_msrs_dependency_persistence.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
LABELING = REPO / "labeling"
PRED_DIR = REPO / "data" / "ib_data" / "predictions"
DIAG_DIR = REPO / "data" / "ib_data" / "diagnostics"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LABELING) not in sys.path:
    sys.path.insert(0, str(LABELING))

from run_plmn import (  # noqa: E402
    Config,
    GRUPredictor,
    device_for_run,
    prepare_unlabeled_plmn,
    residual_matrix,
)
from tool import human_anomaly_mask, load_labels  # noqa: E402


def _torch_load(path: Path) -> dict[str, Any]:
    """Load checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(m):
        if not m[i]:
            i += 1
            continue
        j = i + 1
        while j < len(m) and m[j]:
            j += 1
        out.append((i, j - 1))
        i = j
    return out


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _robust_center_scale(x: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x, axis=axis)
    mad = np.nanmedian(np.abs(x - np.expand_dims(med, axis=axis)), axis=axis)
    scale = 1.4826 * mad
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1e-8)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.asarray(med, dtype=np.float64), np.asarray(scale, dtype=np.float64)


def _robust_standardize(
    train: np.ndarray,
    valid: np.ndarray,
    *,
    clip: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    med = np.median(train, axis=0)
    mad = np.median(np.abs(train - med[None, :]), axis=0)
    scale = 1.4826 * mad
    # Constant / near-constant counters remain usable without division blow-up.
    fallback = np.std(train, axis=0)
    scale = np.where(scale > 1e-8, scale, np.maximum(fallback, 1e-8))
    train_z = np.clip((train - med[None, :]) / scale[None, :], -clip, clip)
    valid_z = np.clip((valid - med[None, :]) / scale[None, :], -clip, clip)
    return train_z, valid_z, med, scale


def dependency_evidence(
    train: np.ndarray,
    valid: np.ndarray,
    *,
    alpha: float,
    clip: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one train-only Ridge relation model per feature.

    Feature f is predicted from all *other* features at the same time step.
    A high valid residual means the current multivariate relationship is poorly
    explained by relationships learned from the unlabeled train split.
    """
    train_z, valid_z, med, scale = _robust_standardize(
        train, valid, clip=clip
    )
    n_train, n_features = train_z.shape
    valid_abs = np.zeros_like(valid_z, dtype=np.float64)
    train_abs = np.zeros_like(train_z, dtype=np.float64)
    coef_l1 = np.zeros(n_features, dtype=np.float64)
    train_r2 = np.zeros(n_features, dtype=np.float64)

    for f in range(n_features):
        keep = np.ones(n_features, dtype=bool)
        keep[f] = False

        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(train_z[:, keep], train_z[:, f])

        pred_train = model.predict(train_z[:, keep])
        pred_valid = model.predict(valid_z[:, keep])
        train_abs[:, f] = np.abs(train_z[:, f] - pred_train)
        valid_abs[:, f] = np.abs(valid_z[:, f] - pred_valid)

        coef_l1[f] = float(np.sum(np.abs(model.coef_)))
        train_r2[f] = float(model.score(train_z[:, keep], train_z[:, f]))

    ref_med = np.median(train_abs, axis=0)
    ref_mad = np.median(np.abs(train_abs - ref_med[None, :]), axis=0)
    ref_scale = 1.4826 * ref_mad
    ref_scale = np.where(ref_scale > 1e-8, ref_scale, 1e-8)

    dep_z = np.maximum(
        (valid_abs - ref_med[None, :]) / ref_scale[None, :],
        0.0,
    )
    dep_evidence = np.log1p(dep_z)

    meta = {
        "ridge_alpha": float(alpha),
        "robust_input_clip": float(clip),
        "n_train_points": int(n_train),
        "train_feature_median": med.tolist(),
        "train_feature_scale": scale.tolist(),
        "relation_residual_median": ref_med.tolist(),
        "relation_residual_scale": ref_scale.tolist(),
        "relation_coef_l1": coef_l1.tolist(),
        "relation_train_r2": train_r2.tolist(),
        "mean_train_r2": float(np.mean(train_r2)),
        "median_train_r2": float(np.median(train_r2)),
    }
    return dep_evidence, meta


def sparse_feature_evidence_from_checkpoint(
    *,
    checkpoint: dict[str, Any],
    prep: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[int, np.ndarray], int]:
    """Recreate per-feature forecast-residual evidence for each MSRS scale."""
    cfg_dict = dict(checkpoint["config"])
    cfg_dict["scales"] = tuple(int(x) for x in cfg_dict["scales"])
    cfg = Config(**cfg_dict)

    n_features = len(prep["cols"])
    k = max(
        1,
        min(
            n_features,
            int(math.ceil(n_features * float(cfg.topk_ratio))),
        ),
    )

    evidence_by_scale: dict[int, np.ndarray] = {}
    for window in cfg.scales:
        model = GRUPredictor(
            n_features=n_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        model.load_state_dict(checkpoint["models"][str(window)])
        model.to(device)
        model.eval()

        valid_resid = residual_matrix(
            model,
            prep["valid"],
            window=int(window),
            batch_size=int(batch_size),
            device=device,
        )

        ref = checkpoint["residual_reference"][str(window)]
        feat_med = np.asarray(ref["median"], dtype=np.float64)
        feat_scale = np.asarray(ref["scale"], dtype=np.float64)
        feat_scale = np.where(feat_scale > 1e-8, feat_scale, 1e-8)

        z = np.maximum(
            (valid_resid - feat_med[None, :]) / feat_scale[None, :],
            0.0,
        )
        evidence_by_scale[int(window)] = np.log1p(z)

    stack = np.stack(
        [evidence_by_scale[w] for w in sorted(evidence_by_scale)],
        axis=0,
    )
    with np.errstate(invalid="ignore"):
        combined = np.nanmax(stack, axis=0)

    max_window = max(cfg.scales)
    combined[:max_window] = np.nan
    return combined, evidence_by_scale, k


def candidate_dependency_score(
    sparse_feature_evidence: np.ndarray,
    dep_evidence: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    """Measure relation violation on the SAME features flagged by sparse scoring."""
    n, f = sparse_feature_evidence.shape
    mean_score = np.full(n, np.nan, dtype=np.float64)
    max_score = np.full(n, np.nan, dtype=np.float64)
    strong_count = np.full(n, np.nan, dtype=np.float64)
    top_indices: list[list[int]] = [[] for _ in range(n)]

    strong_cut = math.log1p(3.0)  # > 3 robust scales above train residual median

    for t in range(n):
        row = sparse_feature_evidence[t]
        if not np.isfinite(row).all():
            continue
        idx = np.argpartition(row, f - k)[-k:]
        idx = idx[np.argsort(row[idx])[::-1]]
        top_indices[t] = [int(x) for x in idx]

        dep = dep_evidence[t, idx]
        mean_score[t] = float(np.mean(dep))
        max_score[t] = float(np.max(dep))
        strong_count[t] = float(np.sum(row > strong_cut))

    return mean_score, max_score, strong_count, top_indices


def causal_persistence(
    scores: np.ndarray,
    *,
    threshold: float,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Causal rolling persistence; only present and past points are used."""
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    ratio = np.full(n, np.nan, dtype=np.float64)
    ok = np.isfinite(s)
    ratio[ok] = s[ok] / max(float(threshold), 1e-12)

    mean_ratio = np.full(n, np.nan, dtype=np.float64)
    fraction_above = np.full(n, np.nan, dtype=np.float64)
    max_ratio = np.full(n, np.nan, dtype=np.float64)

    for t in range(n):
        a = max(0, t - horizon + 1)
        w = ratio[a : t + 1]
        finite = w[np.isfinite(w)]
        if finite.size == 0:
            continue
        mean_ratio[t] = float(np.mean(finite))
        fraction_above[t] = float(np.mean(finite > 1.0))
        max_ratio[t] = float(np.max(finite))

    return {
        "score_ratio": ratio,
        "mean_ratio": mean_ratio,
        "fraction_above_threshold": fraction_above,
        "max_ratio": max_ratio,
    }


def _finite_summary(x: list[float | None]) -> dict[str, float | int | None]:
    arr = np.asarray(
        [v for v in x if v is not None and np.isfinite(v)],
        dtype=np.float64,
    )
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _peak_index(scores: np.ndarray, a: int, b: int) -> int:
    part = scores[a : b + 1]
    finite = np.isfinite(part)
    if not finite.any():
        return int(a)
    local = np.where(finite, part, -np.inf)
    return int(a + np.argmax(local))


def build_segment_row(
    *,
    category: str,
    event_id: int,
    seg: tuple[int, int],
    times: pd.Series,
    scores: np.ndarray,
    threshold: float,
    dep_mean: np.ndarray,
    dep_max: np.ndarray,
    strong_count: np.ndarray,
    persistence: dict[str, np.ndarray],
    top_indices: list[list[int]],
    sparse_feature_evidence: np.ndarray,
    dep_evidence: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    a, b = seg
    peak = _peak_index(scores, a, b)
    idx = top_indices[peak]

    top_features = [feature_names[i] for i in idx]
    top_sparse = [
        float(sparse_feature_evidence[peak, i])
        if np.isfinite(sparse_feature_evidence[peak, i])
        else None
        for i in idx
    ]
    top_dep = [
        float(dep_evidence[peak, i])
        if np.isfinite(dep_evidence[peak, i])
        else None
        for i in idx
    ]

    def seg_max(arr: np.ndarray) -> float | None:
        x = arr[a : b + 1]
        x = x[np.isfinite(x)]
        return float(np.max(x)) if x.size else None

    def seg_mean(arr: np.ndarray) -> float | None:
        x = arr[a : b + 1]
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else None

    return {
        "category": category,
        "event_id": int(event_id),
        "start_index": int(a),
        "end_index": int(b),
        "length_points": int(b - a + 1),
        "start_time": pd.Timestamp(times.iloc[a]).isoformat(),
        "end_time": pd.Timestamp(times.iloc[b]).isoformat(),
        "peak_time": pd.Timestamp(times.iloc[peak]).isoformat(),
        "max_msrs_score": seg_max(scores),
        "mean_msrs_score": seg_mean(scores),
        "max_score_ratio": seg_max(persistence["score_ratio"]),
        "max_dependency_mean": seg_max(dep_mean),
        "mean_dependency_mean": seg_mean(dep_mean),
        "max_dependency_max": seg_max(dep_max),
        "max_persistence_mean_ratio": seg_max(persistence["mean_ratio"]),
        "mean_persistence_mean_ratio": seg_mean(persistence["mean_ratio"]),
        "max_persistence_fraction": seg_max(
            persistence["fraction_above_threshold"]
        ),
        "max_strong_sparse_feature_count": seg_max(strong_count),
        "peak_top_features": top_features,
        "peak_top_sparse_evidence": top_sparse,
        "peak_top_dependency_evidence": top_dep,
        "threshold": float(threshold),
    }


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "max_msrs_score",
        "max_score_ratio",
        "max_dependency_mean",
        "max_dependency_max",
        "max_persistence_mean_ratio",
        "max_persistence_fraction",
        "max_strong_sparse_feature_count",
    )
    return {
        field: _finite_summary([row.get(field) for row in rows])
        for field in fields
    }


def separation_auc(
    gt_rows: list[dict[str, Any]],
    fa_rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    """AUC > .5 means the diagnostic tends to be larger for GT than false alarms."""
    fields = (
        "max_dependency_mean",
        "max_dependency_max",
        "max_persistence_mean_ratio",
        "max_persistence_fraction",
        "max_strong_sparse_feature_count",
    )
    out: dict[str, float | None] = {}
    all_rows = [(1, r) for r in gt_rows] + [(0, r) for r in fa_rows]

    for field in fields:
        y: list[int] = []
        s: list[float] = []
        for label, row in all_rows:
            value = row.get(field)
            if value is None or not np.isfinite(value):
                continue
            y.append(label)
            s.append(float(value))
        if len(set(y)) < 2:
            out[field] = None
        else:
            out[field] = float(roc_auc_score(y, s))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", default="P0480")
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--relation_clip", type=float, default=8.0)
    p.add_argument(
        "--persistence_horizon",
        type=int,
        default=6,
        help="causal rolling horizon in points; 6 = 30 min at 5-min sampling",
    )
    p.add_argument("--batch_size", type=int, default=512)
    args = p.parse_args()

    pred_path = PRED_DIR / f"{args.plmn}_msrs_valid.json"
    ckpt_path = ROOT / "model" / args.plmn / "paperib" / "msrs.pt"
    if not pred_path.exists():
        raise SystemExit(
            f"missing {pred_path}; run MSRS/run_plmn.py first"
        )
    if not ckpt_path.exists():
        raise SystemExit(
            f"missing {ckpt_path}; run MSRS/run_plmn.py locally first"
        )

    pred_doc = json.loads(pred_path.read_text(encoding="utf-8"))
    checkpoint = _torch_load(ckpt_path)
    prep = prepare_unlabeled_plmn(args.plmn)

    scores = np.asarray(
        [
            np.nan if x is None else float(x)
            for x in pred_doc["score_series"]["scores"]
        ],
        dtype=np.float64,
    )
    threshold = float(pred_doc["threshold"])
    times_valid = prep["times"].iloc[
        prep["train_end"] : prep["valid_end"]
    ].reset_index(drop=True)

    if len(scores) != len(prep["valid"]):
        raise RuntimeError(
            f"prediction length {len(scores)} != valid length {len(prep['valid'])}"
        )

    device = device_for_run()
    print(
        f"device={device} plmn={args.plmn} valid_points={len(scores)} "
        f"features={len(prep['cols'])}",
        flush=True,
    )
    print(
        "building train-only sparse feature evidence and dependency models",
        flush=True,
    )

    sparse_feature_evidence, _by_scale, k = (
        sparse_feature_evidence_from_checkpoint(
            checkpoint=checkpoint,
            prep=prep,
            device=device,
            batch_size=args.batch_size,
        )
    )

    dep_evidence, dep_meta = dependency_evidence(
        prep["train"],
        prep["valid"],
        alpha=args.ridge_alpha,
        clip=args.relation_clip,
    )

    dep_mean, dep_max, strong_count, top_indices = (
        candidate_dependency_score(
            sparse_feature_evidence,
            dep_evidence,
            k=k,
        )
    )

    persistence = causal_persistence(
        scores,
        threshold=threshold,
        horizon=args.persistence_horizon,
    )

    # Strict separation: labels are loaded only after all train-only diagnostic
    # features have been fitted and valid diagnostic signals have been computed.
    print(
        "diagnostic signals complete; loading expert labels for comparison only",
        flush=True,
    )
    labels = load_labels(args.plmn)
    y_full = human_anomaly_mask(
        prep["df"],
        labels.get("labels"),
    ).astype(bool)
    y_valid = y_full[prep["train_end"] : prep["valid_end"]]

    pred_mask = np.isfinite(scores) & (scores > threshold)
    gt_segments = _segments(y_valid)
    pred_segments = _segments(pred_mask)
    fa_segments = [
        seg
        for seg in pred_segments
        if not any(_overlap(seg, gt) for gt in gt_segments)
    ]

    gt_rows = [
        build_segment_row(
            category="gt_event",
            event_id=i,
            seg=seg,
            times=times_valid,
            scores=scores,
            threshold=threshold,
            dep_mean=dep_mean,
            dep_max=dep_max,
            strong_count=strong_count,
            persistence=persistence,
            top_indices=top_indices,
            sparse_feature_evidence=sparse_feature_evidence,
            dep_evidence=dep_evidence,
            feature_names=prep["cols"],
        )
        for i, seg in enumerate(gt_segments, start=1)
    ]
    fa_rows = [
        build_segment_row(
            category="false_alarm",
            event_id=i,
            seg=seg,
            times=times_valid,
            scores=scores,
            threshold=threshold,
            dep_mean=dep_mean,
            dep_max=dep_max,
            strong_count=strong_count,
            persistence=persistence,
            top_indices=top_indices,
            sparse_feature_evidence=sparse_feature_evidence,
            dep_evidence=dep_evidence,
            feature_names=prep["cols"],
        )
        for i, seg in enumerate(fa_segments, start=1)
    ]

    aucs = separation_auc(gt_rows, fa_rows)
    result = {
        "plmn": args.plmn,
        "source_prediction": str(pred_path.relative_to(REPO)),
        "checkpoint": str(ckpt_path.relative_to(REPO)),
        "train_labels_used": False,
        "validation_labels_used_for_diagnostic_fitting": False,
        "n_gt_events": len(gt_rows),
        "n_false_alarm_events": len(fa_rows),
        "topk_features": int(k),
        "persistence_horizon_points": int(args.persistence_horizon),
        "persistence_horizon_minutes": int(args.persistence_horizon * 5),
        "dependency_method": {
            "type": "per-feature Ridge from all other contemporaneous features",
            **dep_meta,
        },
        "group_summary": {
            "gt_event": group_summary(gt_rows),
            "false_alarm": group_summary(fa_rows),
        },
        "separation_auc_gt_vs_false_alarm": aucs,
        "interpretation": {
            "auc_rule": (
                "AUC > 0.5 means the diagnostic tends to be larger for GT "
                "events than current MSRS false alarms; values near 0.5 mean "
                "little separation."
            ),
            "dependency": (
                "If dependency AUC is clearly > 0.5, a relation-verification "
                "branch is justified."
            ),
            "persistence": (
                "If persistence AUC is clearly > 0.5, temporal persistence "
                "gating is justified."
            ),
            "sparse_count": (
                "If sparse-feature-count AUC is > 0.5, GT events tend to affect "
                "more variables than false alarms; if < 0.5, false alarms may "
                "be broader than GT."
            ),
        },
        "gt_events": gt_rows,
        "false_alarm_events": fa_rows,
    }

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = (
        DIAG_DIR
        / f"{args.plmn}_msrs_dependency_persistence.json"
    )
    csv_path = (
        DIAG_DIR
        / f"{args.plmn}_msrs_dependency_persistence.csv"
    )
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = gt_rows + fa_rows
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = dict(row)
        flat["peak_top_features"] = "|".join(row["peak_top_features"])
        flat["peak_top_sparse_evidence"] = "|".join(
            "" if x is None else f"{x:.6g}"
            for x in row["peak_top_sparse_evidence"]
        )
        flat["peak_top_dependency_evidence"] = "|".join(
            "" if x is None else f"{x:.6g}"
            for x in row["peak_top_dependency_evidence"]
        )
        csv_rows.append(flat)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"GT events={len(gt_rows)} false alarms={len(fa_rows)}", flush=True)
    print("GT-vs-FA separation AUC:", flush=True)
    for key, value in aucs.items():
        text = "None" if value is None else f"{value:.4f}"
        print(f"  {key}: {text}", flush=True)

    print("\nGroup medians:", flush=True)
    for group in ("gt_event", "false_alarm"):
        summary = result["group_summary"][group]
        print(f"  [{group}]", flush=True)
        for key in (
            "max_dependency_mean",
            "max_dependency_max",
            "max_persistence_mean_ratio",
            "max_persistence_fraction",
            "max_strong_sparse_feature_count",
        ):
            print(
                f"    {key}: {summary[key]['median']}",
                flush=True,
            )

    print(f"\nwrote {json_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
