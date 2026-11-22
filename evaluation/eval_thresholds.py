#!/usr/bin/env python3
"""ContextAE를 다시 학습하지 않고 operational threshold 규칙들을 비교하는 스크립트.

각 temporal context의 Final clean TRAIN reconstruction-error 분포만 이용해
threshold를 계산하고, VALID 전문가 label은 결과 평가에만 사용한다.

비교 방법:
- 현재 방식인 mean + k*sigma
- TRAIN empirical quantile
- robust median + k * 1.4826 * MAD
- POT/EVT의 GPD tail fit

결과는 각 ContextAE validation prediction JSON의 threshold_evaluation에 저장하고,
results/contextae_threshold_summary.csv/json에도 요약한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import genpareto

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.metrics import detection_delay, event_f1, false_alarms_per_day, point_prf

PRED_DIR = ROOT / "data" / "ib_data" / "predictions"
LABEL_DIR = ROOT / "data" / "ib_data" / "labels"
MODEL_DIR = ROOT / "ContextAE" / "model"
RESULT_DIR = ROOT / "results"

from core.features import PAPER_PLMNS


from models.contextae import run_plmn as CTX



def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _label_mask(times: pd.DatetimeIndex, label_doc: dict[str, Any]) -> np.ndarray:
    y = np.zeros(len(times), dtype=bool)
    for item in label_doc.get("labels") or []:
        start = item.get("start")
        end = item.get("end") or start
        if not start:
            continue
        a = pd.Timestamp(start)
        b = pd.Timestamp(end)
        if a.tzinfo is None:
            a = a.tz_localize("UTC")
        else:
            a = a.tz_convert("UTC")
        if b.tzinfo is None:
            b = b.tz_localize("UTC")
        else:
            b = b.tz_convert("UTC")
        y |= (times >= a) & (times <= b)
    return y


def _slot_map(segmentation: dict[str, Any]) -> np.ndarray:
    mapping = segmentation.get("slot_to_context") or {}
    if len(mapping) != 288:
        raise ValueError("checkpoint segmentation.slot_to_context must contain 288 slots")
    return np.asarray(
        [int(mapping[CTX.slot_label(i)]) for i in range(288)],
        dtype=np.int16,
    )


def _model_from_state(
    state: dict[str, torch.Tensor],
    *,
    n_features: int,
    cfg: Any,
) -> torch.nn.Module:
    model = CTX.PointAutoEncoder(
        n_features=n_features,
        hidden_dim=int(cfg.hidden_dim),
        latent_dim=int(cfg.latent_dim),
    )
    model.load_state_dict(state)
    model.to(torch.device("cpu"))
    model.eval()
    return model


def _clean_train_scores(
    plmn: str,
    checkpoint: dict[str, Any],
) -> tuple[dict[int, np.ndarray], Any]:
    cfg_dict = dict(checkpoint.get("config") or {})
    cfg = CTX.Config(**cfg_dict)
    prep = CTX.prepare_unlabeled_plmn(plmn, cfg.timezone)

    ckpt_cols = list(checkpoint.get("feature_columns") or [])
    if ckpt_cols and ckpt_cols != list(prep["cols"]):
        raise ValueError(
            "checkpoint feature columns differ from current data feature columns"
        )

    slot_to_context = _slot_map(checkpoint["segmentation"])
    train_context_ids = CTX.context_ids_from_slots(
        prep["train_slots"],
        slot_to_context,
    )
    n_features = len(prep["cols"])
    out: dict[int, np.ndarray] = {}

    for key, item in (checkpoint.get("contexts") or {}).items():
        cid = int(key)
        idx = np.flatnonzero(train_context_ids == cid)
        if idx.size == 0:
            continue

        raw = prep["train"][idx]
        stage1_model = _model_from_state(
            item["stage1_state"],
            n_features=n_features,
            cfg=cfg,
        )
        x1 = item["stage1_scaler"].transform(raw).astype(np.float32)
        stage1_scores = CTX.reconstruction_mse(
            stage1_model,
            x1,
            batch_size=int(cfg.batch_size),
            device=torch.device("cpu"),
        )
        clean = stage1_scores <= float(item["stage1_threshold_mse"])
        if int(clean.sum()) < 8:
            raise ValueError(f"{plmn} context {cid}: too few clean train samples")

        final_model = _model_from_state(
            item["final_state"],
            n_features=n_features,
            cfg=cfg,
        )
        x2 = item["final_scaler"].transform(raw[clean]).astype(np.float32)
        scores = CTX.reconstruction_mse(
            final_model,
            x2,
            batch_size=int(cfg.batch_size),
            device=torch.device("cpu"),
        )
        scores = np.asarray(scores, dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if scores.size < 8:
            raise ValueError(f"{plmn} context {cid}: no finite clean train scores")
        out[cid] = scores

    return out, cfg


def _mad_threshold(scores: np.ndarray, k: float) -> float:
    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma < 1e-12:
        robust_sigma = float(np.std(scores, ddof=0))
    if not np.isfinite(robust_sigma) or robust_sigma < 1e-12:
        robust_sigma = 1e-12
    return float(med + float(k) * robust_sigma)


def _pot_threshold(
    scores: np.ndarray,
    *,
    u_quantile: float,
    alpha: float,
    min_exceedances: int,
) -> tuple[float, dict[str, Any]]:
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = int(s.size)
    if n < 8:
        raise ValueError("too few train scores for POT")

    u = float(np.quantile(s, u_quantile))
    excess = s[s > u] - u
    p_u = float(len(excess) / n)

    fallback_q = float(np.clip(1.0 - alpha, 0.0, 1.0))
    fallback = float(np.quantile(s, fallback_q))

    meta: dict[str, Any] = {
        "u_quantile": float(u_quantile),
        "u": u,
        "alpha": float(alpha),
        "n_train": n,
        "n_exceedances": int(len(excess)),
        "exceedance_fraction": p_u,
        "fallback": False,
    }

    if len(excess) < int(min_exceedances) or alpha <= 0 or alpha >= p_u:
        meta["fallback"] = True
        meta["fallback_reason"] = "insufficient_tail_or_invalid_alpha"
        return fallback, meta

    try:
        shape, loc, scale = genpareto.fit(excess, floc=0.0)
        if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("invalid GPD fit")
        conditional_cdf = 1.0 - float(alpha / p_u)
        conditional_cdf = float(np.clip(conditional_cdf, 0.0, 1.0 - 1e-12))
        y = float(
            genpareto.ppf(
                conditional_cdf,
                c=float(shape),
                loc=0.0,
                scale=float(scale),
            )
        )
        threshold = float(u + y)
        if not np.isfinite(threshold):
            raise ValueError("non-finite POT threshold")
        meta.update(
            {
                "gpd_shape": float(shape),
                "gpd_scale": float(scale),
                "conditional_cdf": conditional_cdf,
            }
        )
        return threshold, meta
    except Exception as exc:
        meta["fallback"] = True
        meta["fallback_reason"] = f"gpd_fit_failed:{type(exc).__name__}"
        return fallback, meta


def _method_specs(args: argparse.Namespace, cfg: Any) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "key": f"sigma_{float(cfg.threshold_sigma):g}",
            "label": f"Mean + {float(cfg.threshold_sigma):g}σ (current)",
            "kind": "sigma",
            "k": float(cfg.threshold_sigma),
        }
    ]
    for q in args.quantiles:
        specs.append(
            {
                "key": f"quantile_{q:g}",
                "label": f"Quantile q={q:g}",
                "kind": "quantile",
                "q": float(q),
            }
        )
    for k in args.mad_k:
        specs.append(
            {
                "key": f"mad_{k:g}",
                "label": f"Median + {k:g}·1.4826MAD",
                "kind": "mad",
                "k": float(k),
            }
        )
    for alpha in args.pot_alpha:
        specs.append(
            {
                "key": f"pot_a{alpha:g}",
                "label": f"POT α={alpha:g}",
                "kind": "pot",
                "alpha": float(alpha),
                "u_quantile": float(args.pot_u_quantile),
                "min_exceedances": int(args.pot_min_exceedances),
            }
        )
    return specs


def _thresholds_for_method(
    clean_scores: dict[int, np.ndarray],
    spec: dict[str, Any],
) -> tuple[dict[int, float], dict[int, dict[str, Any]]]:
    thresholds: dict[int, float] = {}
    details: dict[int, dict[str, Any]] = {}

    for cid, scores in clean_scores.items():
        kind = spec["kind"]
        if kind == "sigma":
            mu = float(np.mean(scores))
            sd = float(np.std(scores, ddof=0))
            if not np.isfinite(sd) or sd < 1e-12:
                sd = 1e-12
            thr = float(mu + float(spec["k"]) * sd)
            meta = {"mean": mu, "std": sd}
        elif kind == "quantile":
            thr = float(np.quantile(scores, float(spec["q"])))
            meta = {"q": float(spec["q"])}
        elif kind == "mad":
            thr = _mad_threshold(scores, float(spec["k"]))
            meta = {
                "median": float(np.median(scores)),
                "mad": float(np.median(np.abs(scores - np.median(scores)))),
                "k": float(spec["k"]),
            }
        elif kind == "pot":
            thr, meta = _pot_threshold(
                scores,
                u_quantile=float(spec["u_quantile"]),
                alpha=float(spec["alpha"]),
                min_exceedances=int(spec["min_exceedances"]),
            )
        else:
            raise ValueError(f"unknown threshold kind: {kind}")
        thresholds[int(cid)] = float(thr)
        details[int(cid)] = meta

    return thresholds, details


def _evaluate_prediction(
    y_true: np.ndarray,
    raw_mse: np.ndarray,
    context_ids: np.ndarray,
    thresholds: dict[int, float],
    *,
    sample_minutes: float,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    raw = np.asarray(raw_mse, dtype=np.float64).reshape(-1)
    ctx = np.asarray(context_ids, dtype=np.int16).reshape(-1)
    n = min(len(y), len(raw), len(ctx))
    y, raw, ctx = y[:n], raw[:n], ctx[:n]

    pred = np.zeros(n, dtype=bool)
    finite = np.isfinite(raw)
    for cid, thr in thresholds.items():
        mask = finite & (ctx == int(cid))
        pred[mask] = raw[mask] > float(thr)

    point = point_prf(y, pred)
    event = event_f1(y, pred)
    fa = false_alarms_per_day(y, pred, sample_minutes=sample_minutes)
    delay = detection_delay(y, pred, sample_minutes=sample_minutes)

    vals = np.asarray(list(thresholds.values()), dtype=np.float64)
    return {
        "event_f1": event.get("f1"),
        "event_precision": event.get("precision"),
        "event_recall": event.get("recall"),
        "n_gt_events": event.get("n_gt_events"),
        "n_pred_events": event.get("n_pred_events"),
        "n_hit_gt": event.get("n_hit_gt"),
        "precision": point.get("precision"),
        "recall": point.get("recall"),
        "f1": point.get("f1"),
        "tp": point.get("tp"),
        "fp": point.get("fp"),
        "fn": point.get("fn"),
        "false_alarms_per_day": fa.get("false_alarms_per_day"),
        "n_false_alarm_events": fa.get("n_false_alarm_events"),
        "mean_detection_delay_min": delay.get("mean_detection_delay_min"),
        "median_detection_delay_min": delay.get("median_detection_delay_min"),
        "n_pred_points": int(pred.sum()),
        "threshold_min": float(np.min(vals)),
        "threshold_median": float(np.median(vals)),
        "threshold_max": float(np.max(vals)),
    }


def evaluate_plmn(plmn: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pred_path = PRED_DIR / f"{plmn}_context_ae_valid.json"
    ckpt_path = MODEL_DIR / plmn / "paperib" / "context_ae.pt"
    label_path = LABEL_DIR / f"{plmn}_labels.json"

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} (ContextAE checkpoint is required; rerun ContextAE locally if missing)"
        )
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    pred_doc = _load_json(pred_path)
    ss = pred_doc.get("score_series") or {}
    times = pd.to_datetime(ss.get("times") or [], utc=True)
    raw_values = ss.get("raw_reconstruction_mse") or []
    raw_mse = np.asarray(
        [np.nan if x is None else float(x) for x in raw_values],
        dtype=np.float64,
    )
    context_ids = np.asarray(ss.get("context_ids") or [], dtype=np.int16)
    n = min(len(times), len(raw_mse), len(context_ids))
    if n == 0:
        raise ValueError(f"{pred_path.name}: missing raw score/context series")
    times = times[:n]
    raw_mse = raw_mse[:n]
    context_ids = context_ids[:n]

    y = _label_mask(times, _load_json(label_path))

    try:
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu",
        )
    clean_scores, cfg = _clean_train_scores(plmn, checkpoint)
    specs = _method_specs(args, cfg)

    methods: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    vus_pr = (
        (pred_doc.get("paper_metrics") or {}).get("vus_pr")
        or (pred_doc.get("metrics") or {}).get("vus_pr")
    )

    for spec in specs:
        thresholds, details = _thresholds_for_method(clean_scores, spec)
        metrics = _evaluate_prediction(
            y,
            raw_mse,
            context_ids,
            thresholds,
            sample_minutes=float(cfg.sample_minutes),
        )
        row = {
            "key": spec["key"],
            "label": spec["label"],
            "kind": spec["kind"],
            "parameters": {k: v for k, v in spec.items() if k not in {"key", "label", "kind"}},
            **metrics,
            "thresholds_by_context": {
                str(k): float(v) for k, v in thresholds.items()
            },
            "fit_details_by_context": {
                str(k): v for k, v in details.items()
            },
        }
        methods.append(row)
        flat_rows.append(
            {
                "plmn": plmn,
                "method": spec["key"],
                "label": spec["label"],
                "kind": spec["kind"],
                "vus_pr": vus_pr,
                **metrics,
            }
        )

    block = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "eval_split": "valid",
        "model": "context_ae",
        "threshold_source": "context-wise final clean TRAIN reconstruction error",
        "validation_labels_used_for_threshold_selection": False,
        "validation_labels_used_for_evaluation_only": True,
        "score_basis": "raw_reconstruction_mse",
        "vus_pr": vus_pr,
        "methods": methods,
        "note": (
            "Method comparisons are validation diagnostics. Do not select and report "
            "a validation-oracle threshold as final TEST performance."
        ),
    }
    pred_doc["threshold_evaluation"] = block
    pred_path.write_text(
        json.dumps(pred_doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return flat_rows, block


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all", action="store_true")
    scope.add_argument("--plmn")
    p.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.99, 0.995, 0.999],
    )
    p.add_argument(
        "--mad_k",
        nargs="+",
        type=float,
        default=[3.0, 4.0, 5.0],
    )
    p.add_argument(
        "--pot_alpha",
        nargs="+",
        type=float,
        default=[0.01, 0.005, 0.001],
        help="target unconditional upper-tail probability",
    )
    p.add_argument(
        "--pot_u_quantile",
        type=float,
        default=0.98,
        help="initial POT tail threshold quantile",
    )
    p.add_argument(
        "--pot_min_exceedances",
        type=int,
        default=20,
    )
    args = p.parse_args()

    for q in args.quantiles:
        if not 0.0 < q < 1.0:
            p.error("--quantiles values must be in (0,1)")
    for k in args.mad_k:
        if k <= 0:
            p.error("--mad_k values must be > 0")
    for a in args.pot_alpha:
        if not 0.0 < a < 1.0:
            p.error("--pot_alpha values must be in (0,1)")
    if not 0.5 < args.pot_u_quantile < 1.0:
        p.error("--pot_u_quantile must be in (0.5,1)")
    if args.pot_min_exceedances < 5:
        p.error("--pot_min_exceedances must be >= 5")

    plmns = PAPER_PLMNS if args.all else [str(args.plmn)]
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for plmn in plmns:
        try:
            plmn_rows, _ = evaluate_plmn(plmn, args)
            rows.extend(plmn_rows)
            print(f"\n[{plmn}]")
            for r in plmn_rows:
                print(
                    f"  {r['label']:<30} "
                    f"Event-F1={r['event_f1']:.4f} "
                    f"Point-F1={r['f1']:.4f} "
                    f"P={r['precision']:.4f} R={r['recall']:.4f} "
                    f"FA/day={r['false_alarms_per_day']:.3f}"
                )
        except Exception as exc:
            msg = f"[ERROR] {plmn}: {exc}"
            print(msg, file=sys.stderr)
            failures.append(msg)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULT_DIR / "contextae_threshold_summary.json"
    csv_path = RESULT_DIR / "contextae_threshold_summary.csv"

    if rows:
        json_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fields = [
            "plmn",
            "method",
            "label",
            "kind",
            "vus_pr",
            "event_f1",
            "event_precision",
            "event_recall",
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "fn",
            "false_alarms_per_day",
            "n_false_alarm_events",
            "mean_detection_delay_min",
            "median_detection_delay_min",
            "n_pred_points",
            "threshold_min",
            "threshold_median",
            "threshold_max",
        ]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsummary CSV:  {csv_path}")
        print(f"summary JSON: {json_path}")

    if failures:
        print(f"completed with {len(failures)} failed PLMN(s)")
    if not rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
