#!/usr/bin/env python3
"""Run Argos-style detection on paperib PLMN; evaluate on **valid**.

Paper: Argos — Agentic Time-Series Anomaly Detection with Autonomous Rule
Generation via LLMs ([arXiv:2501.14170](https://arxiv.org/abs/2501.14170),
[microsoft/argos](https://github.com/microsoft/argos)).

Modes
-----
``heuristic``
    No LLM. Fit Argos-format ``inference()`` rules on paperib **train**,
    evaluate on paperib **valid**, write prediction JSON for the labeling UI.
    Default: **multi-metric ensemble** (A_RATE + M971 + fail candidates).

``train-LLM-only`` / ``train-LLM-only-parallel`` / ``train-evolution``
    Export CSV then invoke upstream ``Argos/driver.py`` (needs OpenAI/Azure
    env — see docs/Argos-IB.md). Prefer ``--metrics`` one-at-a-time or
    ensemble of heuristics; full LLM ensemble is expensive.

Examples
--------
::

    # Multi-metric ensemble (default metrics, OR fuse)
    python Argos_ib/run_plmn.py --plmn P0480 --mode heuristic

    # Custom axes + sum threshold (>=2 axes)
    python Argos_ib/run_plmn.py --plmn P0480 --mode heuristic \\
        --metrics A_RATE M971 M520 --fuse sum --score-threshold 2

    # Single metric (legacy)
    python Argos_ib/run_plmn.py --plmn P0480 --mode heuristic --metric A_RATE --no-ensemble
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_THIS = Path(__file__).resolve().parent
_LABELING = _REPO / "labeling"
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
if str(_LABELING) not in sys.path:
    sys.path.insert(0, str(_LABELING))

from apply_rule import event_metrics, point_metrics, run_inference  # noqa: E402
from ensemble import (  # noqa: E402
    DEFAULT_ENSEMBLE_METRICS,
    FUSE_MODES,
    FUSE_OR,
    ensemble_metrics_with_fails,
    fuse_predictions,
    resolve_ensemble_metrics,
)
from export_csv import DEFAULT_OUT, export_plmn_metric  # noqa: E402
from export_predictions import build_prediction_doc, write_prediction_json  # noqa: E402
from heuristic_rules import fit_rule_for_metric, write_rule  # noqa: E402
from tool import load_plmn  # noqa: E402

ARGOS_ROOT = _REPO / "Argos"
PRED_DIR = _REPO / "data" / "ib_data" / "predictions"
RESULT_ROOT = _REPO / "data" / "ib_data" / "argos" / "results"


def _load_csv_splits(csv_path: Path, train_test_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    n = len(df)
    cut = int(n * train_test_split)
    train_df = df.iloc[:cut].copy()
    valid_df = df.iloc[cut:].copy()
    train_df["index"] = range(len(train_df))
    valid_df["index"] = range(len(valid_df))
    return train_df, valid_df


def _fit_heuristic_axis(
    plmn: str,
    metric: str,
    *,
    out_dir: Path,
    stamp: str,
) -> dict[str, Any]:
    meta = export_plmn_metric(plmn, metric, out_dir=out_dir)
    csv_path = Path(meta["csv_path"])
    train_df, valid_df = _load_csv_splits(csv_path, meta["train_test_split"])
    code, fit_meta = fit_rule_for_metric(metric, train_df)
    rule_dir = RESULT_ROOT / plmn / metric / f"heuristic_{stamp}"
    rule_path = write_rule(rule_dir / "rule.py", code)
    (rule_dir / "fit_meta.json").write_text(
        json.dumps(fit_meta, indent=2), encoding="utf-8"
    )
    pred = run_inference(rule_path, valid_df)
    y = valid_df["label"].to_numpy()
    return {
        "metric": metric,
        "pred": pred.astype(np.int32),
        "y": y.astype(np.int32),
        "rule_path": str(rule_path),
        "fit": fit_meta,
        "point": point_metrics(y, pred),
        "event": event_metrics(y, pred),
        "train_test_split": meta["train_test_split"],
        "meta": meta,
    }


def run_heuristic(plmn: str, metric: str, *, out_dir: Path) -> dict:
    """Single-metric heuristic (legacy)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    axis = _fit_heuristic_axis(plmn, metric, out_dir=out_dir, stamp=stamp)
    metrics = {
        "eval_split": "valid",
        "mode": "heuristic",
        "metric": metric,
        "ensemble": False,
        "point": axis["point"],
        "event": axis["event"],
        "fit": axis["fit"],
        "train_test_split": axis["train_test_split"],
    }
    rule_dir = Path(axis["rule_path"]).parent
    (rule_dir / "valid_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    doc = build_prediction_doc(
        plmn,
        metric=metric,
        pred_valid=axis["pred"],
        metrics=metrics,
        rule_path=axis["rule_path"],
        source="argos_heuristic",
    )
    pred_path = PRED_DIR / f"{plmn}_argos_heuristic_{metric}.json"
    write_prediction_json(doc, pred_path)
    metrics["prediction_json"] = str(pred_path)
    metrics["rule_path"] = axis["rule_path"]
    return metrics


def run_heuristic_ensemble(
    plmn: str,
    metrics: list[str],
    *,
    out_dir: Path,
    fuse: str = FUSE_OR,
    score_threshold: int = 1,
) -> dict:
    """Fit one heuristic rule per metric, fuse → valid metrics + UI JSON."""
    df = load_plmn(plmn)
    metrics = resolve_ensemble_metrics(metrics, available=df.columns)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ens_dir = RESULT_ROOT / plmn / "ensemble" / f"heuristic_{stamp}"
    ens_dir.mkdir(parents=True, exist_ok=True)

    axis_preds: dict[str, np.ndarray] = {}
    per_axis: dict[str, Any] = {}
    y_ref: np.ndarray | None = None
    rule_paths: dict[str, str] = {}

    for metric in metrics:
        print(f"[ensemble] fitting {metric} …", flush=True)
        try:
            axis = _fit_heuristic_axis(plmn, metric, out_dir=out_dir, stamp=stamp)
        except SystemExit as exc:
            print(f"[ensemble] skip {metric}: {exc}", flush=True)
            continue
        axis_preds[metric] = axis["pred"]
        rule_paths[metric] = axis["rule_path"]
        per_axis[metric] = {
            "point": axis["point"],
            "event": axis["event"],
            "fit": axis["fit"],
            "rule_path": axis["rule_path"],
        }
        if y_ref is None:
            y_ref = axis["y"]
        elif len(axis["y"]) != len(y_ref):
            raise SystemExit(
                f"label length mismatch on {metric}: {len(axis['y'])} vs {len(y_ref)}"
            )

    if not axis_preds:
        raise SystemExit(f"{plmn}: ensemble produced no axes (check --metrics)")

    fused = fuse_predictions(
        axis_preds, mode=fuse, score_threshold=score_threshold
    )
    pred = fused["pred"]
    # UI convention: lower score = more anomalous → store −(#axes firing).
    score_ui = (-fused["score"]).astype(np.float64)
    assert y_ref is not None
    point = point_metrics(y_ref, pred)
    event = event_metrics(y_ref, pred)

    metrics_out: dict[str, Any] = {
        "eval_split": "valid",
        "mode": "heuristic",
        "ensemble": True,
        "metric": "ensemble",
        "ensemble_metrics": list(axis_preds.keys()),
        "fuse": fused["mode"],
        "score_threshold": fused["score_threshold"],
        "fuse_detail": fused["detail"],
        "per_axis_pred_points": fused["per_axis_pred_points"],
        "per_axis": per_axis,
        "point": point,
        "event": event,
        "train_test_split": None,
    }
    (ens_dir / "valid_metrics.json").write_text(
        json.dumps(metrics_out, indent=2, default=str), encoding="utf-8"
    )
    np.savez_compressed(
        ens_dir / "axis_preds.npz",
        **{k: v for k, v in axis_preds.items()},
        __score=fused["score"],
        __pred=pred,
        __y=y_ref,
    )
    (ens_dir / "rule_paths.json").write_text(
        json.dumps(rule_paths, indent=2), encoding="utf-8"
    )

    doc = build_prediction_doc(
        plmn,
        metric="ensemble",
        pred_valid=pred,
        metrics=metrics_out,
        rule_path=str(ens_dir),
        source="argos_heuristic_ensemble",
        score_valid=score_ui,
        ensemble_metrics=list(axis_preds.keys()),
    )
    # Align score_series threshold with −(#fires) convention.
    ss = doc.get("score_series") or {}
    ss["threshold"] = float(-(max(1, score_threshold) - 0.5))
    ss["score_scale"] = "neg_axis_count"
    doc["score_series"] = ss
    doc["threshold"] = ss["threshold"]

    pred_path = PRED_DIR / f"{plmn}_argos_heuristic_ensemble.json"
    write_prediction_json(doc, pred_path)
    metrics_out["prediction_json"] = str(pred_path)
    metrics_out["ensemble_dir"] = str(ens_dir)
    metrics_out["rule_paths"] = rule_paths
    return metrics_out


def run_argos_llm(
    plmn: str,
    metric: str,
    *,
    mode: str,
    out_dir: Path,
    chunk_size: int,
    top_k: int,
    max_iter: int | None,
    timeout_min: int,
    llm_engine: str = "gpt-4o",
) -> dict:
    if not (ARGOS_ROOT / "driver.py").is_file():
        raise SystemExit(
            f"Upstream Argos not found at {ARGOS_ROOT}.\n"
            "Clone: git clone https://github.com/microsoft/argos.git Argos"
        )

    meta = export_plmn_metric(plmn, metric, out_dir=out_dir)
    csv_path = Path(meta["csv_path"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULT_ROOT / plmn / metric / f"{mode}_{stamp}"
    result_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ARGOS_ROOT / "driver.py"),
        f"--dataset_path={csv_path}",
        f"--mode={mode}",
        f"--result_path={result_path}",
        f"--chunk_size={chunk_size}",
        f"--top_k={top_k}",
        "--dataset_mode=one-by-one",
        f"--train_test_split={meta['train_test_split']}",
        f"--timeout={timeout_min}",
        f"--llm_engine={llm_engine}",
    ]
    if max_iter is not None:
        cmd.append(f"--max_iter={max_iter}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ARGOS_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print("Running:", " ".join(cmd), flush=True)
    print(f"LLM engine: {llm_engine}", flush=True)
    subprocess.run(cmd, cwd=str(ARGOS_ROOT), env=env, check=True)

    rules = sorted(result_path.rglob("rule*.py"))
    if not rules:
        print("Argos finished but no rule*.py found; skip local valid metrics", flush=True)
        return {"mode": mode, "result_path": str(result_path), "meta": meta}

    rule_path = max(rules, key=lambda p: p.stat().st_mtime)
    train_df, valid_df = _load_csv_splits(csv_path, meta["train_test_split"])
    pred = run_inference(rule_path, valid_df, chunk_size=chunk_size)
    y = valid_df["label"].to_numpy()
    point = point_metrics(y, pred)
    event = event_metrics(y, pred)
    metrics = {
        "eval_split": "valid",
        "mode": mode,
        "metric": metric,
        "ensemble": False,
        "point": point,
        "event": event,
        "rule_path": str(rule_path),
        "argos_result_path": str(result_path),
        "train_test_split": meta["train_test_split"],
    }
    (result_path / "valid_metrics_paperib.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    doc = build_prediction_doc(
        plmn,
        metric=metric,
        pred_valid=pred,
        metrics=metrics,
        rule_path=str(rule_path),
        source="argos",
    )
    pred_path = PRED_DIR / f"{plmn}_argos_{metric}.json"
    write_prediction_json(doc, pred_path)
    metrics["prediction_json"] = str(pred_path)
    return metrics


def run_llm_ensemble(
    plmn: str,
    metrics: list[str],
    *,
    mode: str,
    out_dir: Path,
    chunk_size: int,
    top_k: int,
    max_iter: int | None,
    timeout_min: int,
    llm_engine: str,
    fuse: str,
    score_threshold: int,
) -> dict:
    """Run upstream Argos per metric, then fuse like heuristic ensemble."""
    df = load_plmn(plmn)
    metrics = resolve_ensemble_metrics(metrics, available=df.columns)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ens_dir = RESULT_ROOT / plmn / "ensemble" / f"{mode}_{stamp}"
    ens_dir.mkdir(parents=True, exist_ok=True)

    axis_preds: dict[str, np.ndarray] = {}
    per_axis: dict[str, Any] = {}
    y_ref: np.ndarray | None = None
    rule_paths: dict[str, str] = {}

    for metric in metrics:
        print(f"[ensemble-llm] {metric} …", flush=True)
        one = run_argos_llm(
            plmn,
            metric,
            mode=mode,
            out_dir=out_dir,
            chunk_size=chunk_size,
            top_k=top_k,
            max_iter=max_iter,
            timeout_min=timeout_min,
            llm_engine=llm_engine,
        )
        rule_path = one.get("rule_path")
        if not rule_path:
            print(f"[ensemble-llm] skip {metric}: no rule", flush=True)
            continue
        meta = export_plmn_metric(plmn, metric, out_dir=out_dir)
        _, valid_df = _load_csv_splits(Path(meta["csv_path"]), meta["train_test_split"])
        pred = run_inference(rule_path, valid_df, chunk_size=chunk_size)
        y = valid_df["label"].to_numpy()
        axis_preds[metric] = pred.astype(np.int32)
        rule_paths[metric] = str(rule_path)
        per_axis[metric] = {
            "point": one.get("point"),
            "event": one.get("event"),
            "rule_path": rule_path,
        }
        if y_ref is None:
            y_ref = y.astype(np.int32)

    if not axis_preds or y_ref is None:
        raise SystemExit(f"{plmn}: LLM ensemble produced no axes")

    fused = fuse_predictions(
        axis_preds, mode=fuse, score_threshold=score_threshold
    )
    pred = fused["pred"]
    score_ui = (-fused["score"]).astype(np.float64)
    point = point_metrics(y_ref, pred)
    event = event_metrics(y_ref, pred)
    metrics_out: dict[str, Any] = {
        "eval_split": "valid",
        "mode": mode,
        "ensemble": True,
        "metric": "ensemble",
        "ensemble_metrics": list(axis_preds.keys()),
        "fuse": fused["mode"],
        "score_threshold": fused["score_threshold"],
        "fuse_detail": fused["detail"],
        "per_axis_pred_points": fused["per_axis_pred_points"],
        "per_axis": per_axis,
        "point": point,
        "event": event,
    }
    (ens_dir / "valid_metrics.json").write_text(
        json.dumps(metrics_out, indent=2, default=str), encoding="utf-8"
    )
    doc = build_prediction_doc(
        plmn,
        metric="ensemble",
        pred_valid=pred,
        metrics=metrics_out,
        rule_path=str(ens_dir),
        source="argos_ensemble",
        score_valid=score_ui,
        ensemble_metrics=list(axis_preds.keys()),
    )
    ss = doc.get("score_series") or {}
    ss["threshold"] = float(-(max(1, score_threshold) - 0.5))
    ss["score_scale"] = "neg_axis_count"
    doc["score_series"] = ss
    doc["threshold"] = ss["threshold"]
    pred_path = PRED_DIR / f"{plmn}_argos_ensemble.json"
    write_prediction_json(doc, pred_path)
    metrics_out["prediction_json"] = str(pred_path)
    metrics_out["ensemble_dir"] = str(ens_dir)
    return metrics_out


def _load_dotenv() -> None:
    env_path = _REPO / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def _require_llm_env() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    needed = (
        "OPENAI_AZURE_ENDPOINT",
        "OPENAI_AZURE_API_KEY",
        "OPENAI_AZURE_API_VERSION",
    )
    missing = [k for k in needed if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            "LLM credentials missing. Put one of the following in repo-root .env:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "  # or Azure OpenAI vars — see docs/Argos-IB.md\n"
            f"Missing Azure vars: {', '.join(missing)}"
        )


def _print_metrics(metrics: dict) -> None:
    print(json.dumps(metrics, indent=2, default=str))
    pt = metrics.get("point") or {}
    ev = metrics.get("event") or {}
    if metrics.get("ensemble"):
        print(
            f"\nensemble fuse={metrics.get('fuse')} "
            f"axes={metrics.get('ensemble_metrics')}"
        )
    if pt:
        print(
            f"\nvalid point: P={pt.get('precision', 0):.3f} "
            f"R={pt.get('recall', 0):.3f} F1={pt.get('f1', 0):.3f} "
            f"(TP={pt.get('tp')} FP={pt.get('fp')} FN={pt.get('fn')})"
        )
    if ev:
        print(
            f"valid event: hit={ev.get('n_hit_events')}/{ev.get('n_gt_events')} "
            f"recall={ev.get('event_recall', 0):.3f} "
            f"pred_events={ev.get('n_pred_events')}"
        )


def main() -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plmn", required=True)
    p.add_argument(
        "--metric",
        default=None,
        help="Single metric (implies --no-ensemble if --metrics not set)",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Ensemble axes (default: "
            + " ".join(DEFAULT_ENSEMBLE_METRICS)
            + "; use --with-fails for rate+attempt+fail candidates)"
        ),
    )
    p.add_argument(
        "--with-fails",
        action="store_true",
        help="Add fail_metrics from human labels (fail_surge) to the ensemble",
    )
    p.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Force single-metric run (use --metric, default A_RATE)",
    )
    p.add_argument(
        "--fuse",
        default=FUSE_OR,
        choices=list(FUSE_MODES),
        help="Ensemble fusion: or | sum | two_stage (default or)",
    )
    p.add_argument(
        "--score-threshold",
        type=int,
        default=1,
        help="For fuse=sum/or: min #axes firing (default 1)",
    )
    p.add_argument(
        "--mode",
        default="heuristic",
        choices=[
            "heuristic",
            "train-LLM-only",
            "train-LLM-only-parallel",
            "train-evolution",
        ],
    )
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--chunk_size", type=int, default=2000)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--max_iter", type=int, default=None)
    p.add_argument("--timeout", type=int, default=60, help="Argos timeout minutes")
    p.add_argument(
        "--llm_engine",
        default=None,
        help="Model name (default: OPENAI_MODEL env or gpt-4o)",
    )
    args = p.parse_args()

    if args.mode != "heuristic":
        _require_llm_env()

    out_dir = Path(args.out_dir)
    # Default = multi-metric ensemble. Single-axis only with --no-ensemble
    # or a lone --metric (without --metrics).
    use_ensemble = not args.no_ensemble and not (
        args.metric is not None and args.metrics is None
    )

    if args.mode == "heuristic":
        if use_ensemble:
            if args.metrics:
                metric_list = args.metrics
            elif args.with_fails:
                metric_list = ensemble_metrics_with_fails(
                    args.plmn, available=load_plmn(args.plmn).columns
                )
            else:
                metric_list = list(DEFAULT_ENSEMBLE_METRICS)
            metrics = run_heuristic_ensemble(
                args.plmn,
                metric_list,
                out_dir=out_dir,
                fuse=args.fuse,
                score_threshold=args.score_threshold,
            )
        else:
            metrics = run_heuristic(
                args.plmn,
                args.metric or "A_RATE",
                out_dir=out_dir,
            )
    else:
        llm_engine = (
            args.llm_engine or os.environ.get("OPENAI_MODEL") or "gpt-4o"
        )
        if use_ensemble:
            if args.metrics:
                metric_list = args.metrics
            elif args.with_fails:
                metric_list = ensemble_metrics_with_fails(
                    args.plmn, available=load_plmn(args.plmn).columns
                )
            else:
                metric_list = list(DEFAULT_ENSEMBLE_METRICS)
            metrics = run_llm_ensemble(
                args.plmn,
                metric_list,
                mode=args.mode,
                out_dir=out_dir,
                chunk_size=args.chunk_size,
                top_k=args.top_k,
                max_iter=args.max_iter,
                timeout_min=args.timeout,
                llm_engine=llm_engine,
                fuse=args.fuse,
                score_threshold=args.score_threshold,
            )
        else:
            metrics = run_argos_llm(
                args.plmn,
                args.metric or "A_RATE",
                mode=args.mode,
                out_dir=out_dir,
                chunk_size=args.chunk_size,
                top_k=args.top_k,
                max_iter=args.max_iter,
                timeout_min=args.timeout,
                llm_engine=llm_engine,
            )

    _print_metrics(metrics)


if __name__ == "__main__":
    main()
