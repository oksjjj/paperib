"""Reuse RiAD paper metrics (VUS-PR / Event-F1 / FA/day)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_RIAD_METRICS = Path(__file__).resolve().parents[1] / "RiAD" / "metrics.py"
_spec = importlib.util.spec_from_file_location("riad_metrics_mod", _RIAD_METRICS)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {_RIAD_METRICS}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
evaluate_valid = _mod.evaluate_riad_valid

__all__ = ["evaluate_valid"]
