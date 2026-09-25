"""Multi-metric Argos ensemble for IB inbound roaming.

Each axis is a univariate Argos/heuristic rule. Axis predictions are fused into
a final binary label and an integer anomaly score (# of axes firing).

See docs/논문-스토리라인.md §3.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

# Shared with OmniAnomaly / Anomaly Transformer (labeling/tool.py).
try:
    from tool import EXCLUDED_TRAIN_METRICS, filter_train_metrics
except ImportError:  # pragma: no cover — path when run as script with sys.path
    import sys
    from pathlib import Path

    _lab = Path(__file__).resolve().parents[1] / "labeling"
    if str(_lab) not in sys.path:
        sys.path.insert(0, str(_lab))
    from tool import EXCLUDED_TRAIN_METRICS, filter_train_metrics

# Core screening axes (labeling principles: rate + attempt).
ENSEMBLE_RATE_METRICS: tuple[str, ...] = ("A_RATE",)
ENSEMBLE_ATTEMPT_METRICS: tuple[str, ...] = ("M971",)

# Fail / TO surge axes that appear in human labels (`fail_metrics` / fail_surge).
# Sourced from data/ib_data/labels/*_labels.json (not the full COMB counter set).
ENSEMBLE_FAIL_METRICS: tuple[str, ...] = (
    "M855",
    "M037",
    "M841",
    "M185",
    "M965",
)

DEFAULT_ENSEMBLE_METRICS: tuple[str, ...] = (
    *ENSEMBLE_RATE_METRICS,
    *ENSEMBLE_ATTEMPT_METRICS,
)

# Full IB set (rate + attempt + fail candidates). Prefer with --fuse two_stage
# or --fuse sum --score-threshold 2; plain OR over all fails is very FP-heavy.
FULL_ENSEMBLE_METRICS: tuple[str, ...] = (
    *DEFAULT_ENSEMBLE_METRICS,
    *ENSEMBLE_FAIL_METRICS,
)

FUSE_OR = "or"
FUSE_SUM = "sum"
FUSE_TWO_STAGE = "two_stage"
FUSE_MODES = (FUSE_OR, FUSE_SUM, FUSE_TWO_STAGE)


def fail_metrics_from_labels(plmn: str | None = None) -> list[str]:
    """Fail axes from human label ``fail_metrics`` (PLMN or all labels).

    Falls back to ``ENSEMBLE_FAIL_METRICS`` when no labels / no fail tags.
    """
    import json
    from pathlib import Path

    label_dir = Path(__file__).resolve().parents[1] / "data" / "ib_data" / "labels"
    paths: list[Path]
    if plmn:
        one = label_dir / f"{plmn}_labels.json"
        paths = [one] if one.is_file() else []
    else:
        paths = sorted(label_dir.glob("*_labels.json")) if label_dir.is_dir() else []

    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for lab in doc.get("labels") or []:
            if lab.get("kind") == "normal":
                continue
            for raw in lab.get("fail_metrics") or []:
                m = str(raw).strip()
                if m and m not in seen:
                    seen.add(m)
                    ordered.append(m)
    return ordered or list(ENSEMBLE_FAIL_METRICS)


def ensemble_metrics_with_fails(
    plmn: str | None = None,
    *,
    available: Iterable[str] | None = None,
) -> list[str]:
    """Core rate/attempt axes + label-derived fail metrics."""
    fails = fail_metrics_from_labels(plmn)
    return resolve_ensemble_metrics(
        [*DEFAULT_ENSEMBLE_METRICS, *fails],
        available=available,
    )


def resolve_ensemble_metrics(
    metrics: Sequence[str] | None,
    *,
    available: Iterable[str] | None = None,
) -> list[str]:
    """Default IB axes, optionally filtered to columns that exist."""
    chosen = list(metrics) if metrics else list(DEFAULT_ENSEMBLE_METRICS)
    # de-dupe, preserve order
    seen: set[str] = set()
    ordered: list[str] = []
    for m in chosen:
        key = str(m).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    if available is not None:
        avail = set(available)
        ordered = [m for m in ordered if m in avail]
    ordered = filter_train_metrics(ordered)
    if not ordered:
        raise ValueError("no ensemble metrics resolved (empty after filter)")
    return ordered


def stack_axis_preds(axis_preds: dict[str, np.ndarray]) -> np.ndarray:
    """Shape ``(n_axes, n_time)`` int 0/1."""
    if not axis_preds:
        raise ValueError("axis_preds is empty")
    mats = []
    n = None
    for name, pred in axis_preds.items():
        a = np.asarray(pred, dtype=int).reshape(-1)
        if n is None:
            n = len(a)
        elif len(a) != n:
            raise ValueError(
                f"axis {name!r} length {len(a)} != {n} (ensemble axes must align)"
            )
        mats.append((a > 0).astype(np.int8))
    return np.stack(mats, axis=0)


def fuse_predictions(
    axis_preds: dict[str, np.ndarray],
    *,
    mode: str = FUSE_OR,
    score_threshold: int = 1,
    rate_metrics: Sequence[str] | None = None,
    attempt_metrics: Sequence[str] | None = None,
    fail_metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fuse per-metric binary preds → final mask + score.

    Modes
    -----
    ``or``
        Any axis fires → anomaly. Score = sum of fires.
    ``sum``
        Score = sum of fires; anomaly if score >= ``score_threshold``.
    ``two_stage``
        Stage-1 = OR(rate ∪ attempt); stage-2 = OR(fail).
        Final = stage-1 AND stage-2 (strict confirm). Score still = total fires.
    """
    mode = (mode or FUSE_OR).strip().lower()
    if mode not in FUSE_MODES:
        raise ValueError(f"unknown fuse mode {mode!r}; choose from {FUSE_MODES}")

    names = list(axis_preds.keys())
    stacked = stack_axis_preds(axis_preds)  # (K, T)
    score = stacked.sum(axis=0).astype(np.int32)

    rate_set = set(rate_metrics or ENSEMBLE_RATE_METRICS)
    attempt_set = set(attempt_metrics or ENSEMBLE_ATTEMPT_METRICS)
    fail_set = set(fail_metrics or ENSEMBLE_FAIL_METRICS)
    # Any axis not in rate/attempt is treated as fail/other for two_stage.
    name_to_i = {n: i for i, n in enumerate(names)}

    def _or_group(group: set[str]) -> np.ndarray:
        idxs = [name_to_i[n] for n in names if n in group]
        if not idxs:
            return np.zeros(stacked.shape[1], dtype=bool)
        return stacked[idxs].any(axis=0)

    if mode == FUSE_OR:
        final = score >= max(1, int(score_threshold))
        detail = {"rule": "any axis (score >= thr)"}
    elif mode == FUSE_SUM:
        thr = max(1, int(score_threshold))
        final = score >= thr
        detail = {"rule": f"sum(axes) >= {thr}"}
    else:  # two_stage
        stage1_names = rate_set | attempt_set
        stage1 = _or_group(stage1_names)
        # Fail group: explicit fail list ∩ present, else all non-stage1 axes.
        present_fail = [n for n in names if n in fail_set]
        if not present_fail:
            present_fail = [n for n in names if n not in stage1_names]
        stage2 = _or_group(set(present_fail))
        final = stage1 & stage2
        detail = {
            "rule": "stage1(rate∪attempt) AND stage2(fail)",
            "stage1_metrics": [n for n in names if n in stage1_names],
            "stage2_metrics": present_fail,
            "n_stage1": int(stage1.sum()),
            "n_stage2": int(stage2.sum()),
            "n_both": int(final.sum()),
        }

    per_axis_counts = {
        n: int(np.asarray(axis_preds[n]).astype(bool).sum()) for n in names
    }
    return {
        "pred": final.astype(np.int32),
        "score": score,
        "mode": mode,
        "score_threshold": int(score_threshold),
        "metrics": names,
        "per_axis_pred_points": per_axis_counts,
        "n_pred_points": int(final.sum()),
        "detail": detail,
    }
