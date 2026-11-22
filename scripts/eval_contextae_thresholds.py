#!/usr/bin/env python3
"""이전 ContextAE threshold 평가 실행 경로 호환 wrapper.

실제 구현은 evaluation.eval_thresholds에 있다.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.eval_thresholds import *  # noqa: F401,F403
from evaluation.eval_thresholds import main as _main

if __name__ == "__main__":
    _main()
