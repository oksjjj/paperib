"""RoCA에서도 ContextAE/HourAE와 동일한 paperib 평가 함수를 사용하기 위한 연결 모듈.

모델마다 metric 구현이 달라지는 것을 막기 위해 실제 계산은
common/metrics.py의 evaluate_valid()를 그대로 재사용한다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_COMMON_METRICS = Path(__file__).resolve().parents[1] / "common" / "metrics.py"
_spec = importlib.util.spec_from_file_location("paperib_metrics_mod", _COMMON_METRICS)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load {_COMMON_METRICS}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
evaluate_valid = _mod.evaluate_valid

__all__ = ["evaluate_valid"]
