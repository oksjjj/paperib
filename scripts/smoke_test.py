#!/usr/bin/env python3
"""리팩터링 후 핵심 import와 최소 forward pass를 빠르게 확인하는 smoke test.

전체 모델을 학습하지 않고 다음만 검사한다.

1. core 모듈 import
2. HourAE / ContextAE point AE forward
3. ContextAE context discovery 함수 import
4. RoCA window 변환과 backbone forward
5. evaluation 모듈 import
6. 연구 계층(core/models/evaluation)의 UI 의존성 금지
7. UI facade import
8. raw 데이터가 있으면 첫 논문 PLMN의 11개 core feature 로딩

실행:
    .venv/bin/python scripts/smoke_test.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check(name: str, fn) -> None:
    print(f"[CHECK] {name}")
    fn()
    print(f"[ OK  ] {name}")


def _assert_research_layers_do_not_import_ui() -> None:
    """core/models/evaluation 계층이 ui 패키지를 참조하지 않는지 정적으로 검사한다."""
    violations: list[str] = []
    for rel in ("core", "models", "evaluation"):
        for path in (ROOT / rel).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module == "ui" or module.startswith("ui."):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno} -> from {module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name
                        if module == "ui" or module.startswith("ui."):
                            violations.append(f"{path.relative_to(ROOT)}:{node.lineno} -> import {module}")
    if violations:
        raise AssertionError("research layer -> ui dependency found:\n" + "\n".join(violations))


def main() -> None:
    from core.features import CORE_FEATURES, PAPER_PLMNS
    from core.data import DATA_DIR, load_plmn
    from core.metrics import point_prf
    from models.hourae.model import PointAutoEncoder as HourAEModel
    from models.contextae.model import PointAutoEncoder as ContextAEModel
    from models.contextae.context import slot_label
    from models.roca.config import RoCAConfig
    from models.roca.dataset import block_to_nct
    from models.roca.official.network import base_Model

    check(
        "core metric",
        lambda: point_prf(
            np.array([False, True]),
            np.array([False, True]),
        ),
    )

    def _hour_forward() -> None:
        model = HourAEModel(
            n_features=len(CORE_FEATURES),
            hidden_dim=32,
            latent_dim=8,
        )
        y = model(torch.zeros(2, len(CORE_FEATURES)))
        assert tuple(y.shape) == (2, len(CORE_FEATURES))

    check("HourAE forward", _hour_forward)

    def _context_forward() -> None:
        model = ContextAEModel(
            n_features=len(CORE_FEATURES),
            hidden_dim=32,
            latent_dim=8,
        )
        y = model(torch.zeros(2, len(CORE_FEATURES)))
        assert tuple(y.shape) == (2, len(CORE_FEATURES))
        assert slot_label(0) == "00:00"
        assert slot_label(287) == "23:55"

    check("ContextAE forward/context helper", _context_forward)

    def _roca_forward() -> None:
        cfg = RoCAConfig(input_channels=len(CORE_FEATURES))
        block = np.zeros((64, len(CORE_FEATURES)), dtype=np.float32)
        windows, ends = block_to_nct(
            block,
            window=cfg.window_size,
            time_step=cfg.time_step,
        )
        assert windows.ndim == 3
        assert len(windows) == len(ends)

        model = base_Model(cfg, torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            z, z_hat = model(torch.from_numpy(windows[:2]).float())
        assert z.shape == z_hat.shape

    check("RoCA window/backbone forward", _roca_forward)

    def _evaluation_imports() -> None:
        import evaluation.eval_vus  # noqa: F401
        import evaluation.eval_thresholds  # noqa: F401

    check("evaluation imports", _evaluation_imports)

    check("research layers do not import UI", _assert_research_layers_do_not_import_ui)

    def _ui_facades() -> None:
        # app.py 자체는 Dash 서버용 startup state를 만들기 때문에 여기서는
        # app이 사용하는 역할별 facade까지만 import한다.
        import ui.data_view  # noqa: F401
        import ui.plotting  # noqa: F401
        import ui.ui_labels  # noqa: F401
        import ui.ui_predictions  # noqa: F401

    check("UI facade imports", _ui_facades)

    if DATA_DIR.exists() and any(DATA_DIR.glob("*.csv")):
        def _data_load() -> None:
            plmn = PAPER_PLMNS[0]
            df = load_plmn(plmn)
            present = [c for c in CORE_FEATURES if c in df.columns]
            if not present:
                raise RuntimeError(f"{plmn}: core feature가 하나도 없습니다.")
            print(
                f"        {plmn}: rows={len(df):,}, "
                f"core_features={len(present)}"
            )

        check("raw PLMN data load", _data_load)
    else:
        print("[SKIP ] raw PLMN data load: data/ib_data/raw/*.csv 없음")

    print("\n모든 smoke test를 통과했습니다.")


if __name__ == "__main__":
    main()
