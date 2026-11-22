#!/usr/bin/env python3
"""RoCA 이전 경로 호환 wrapper.

실제 구현은 models.roca.run_plmn에 있다.
기존 실행 명령과 외부 import 호환성을 위해 이 파일을 남겨둔다.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.roca.run_plmn import *  # noqa: F401,F403
from models.roca.run_plmn import main as _main


if __name__ == "__main__":
    _main()
