# scripts/

인바운드 로밍 데이터 준비 및 실행 검증에 사용하는 스크립트입니다. 모든 명령은 저장소 루트에서 실행합니다.

| 파일 | 용도 |
|---|---|
| `export_paper_raw.py` | 전체 비식별화 CSV에서 논문 대상 PLMN만 추출 |
| `build_ib_cache.py` | 원본 CSV로부터 공통 PLMN 캐시 및 순위 생성 |
| `smoke_test.py` | 핵심 모듈 import, 모델 forward, 데이터 로딩 및 의존성 검사 |

```bash
# 전체 사업자 데이터가 별도로 있을 때만 실행
.venv/bin/python scripts/export_paper_raw.py

# IB 공통 캐시 생성 및 검증
.venv/bin/python scripts/build_ib_cache.py
.venv/bin/python scripts/smoke_test.py
```

## VUS 재평가

저장된 validation score JSON을 사용해 모델 재학습 없이 VUS 지표를 계산합니다.

```bash
.venv/bin/python -m evaluation.eval_vus --all
```

모델을 제한하려면:

```bash
.venv/bin/python -m evaluation.eval_vus --all --models context_ae hour_ae roca
```

결과는 각 prediction JSON의 `paper_metrics` / `metrics`에 기록되고,
`results/vus_summary.csv`와 `results/vus_summary.json`에도 저장됩니다.


## ContextAE threshold 비교

기존 ContextAE checkpoint를 재사용하여 재학습 없이 operational threshold를 비교합니다.

비교 방법:
- 현재 `mean + kσ`
- train quantile
- `median + k·1.4826·MAD`
- POT/EVT (GPD)

10개 사업자 전체:

```bash
.venv/bin/python -m evaluation.eval_thresholds --all
```

한 사업자만:

```bash
.venv/bin/python -m evaluation.eval_thresholds --plmn P0480
```

기본 후보를 직접 지정하려면:

```bash
.venv/bin/python -m evaluation.eval_thresholds \
  --all \
  --quantiles 0.99 0.995 0.999 \
  --mad_k 3 4 5 \
  --pot_alpha 0.01 0.005 0.001 \
  --pot_u_quantile 0.98
```

결과:
- 각 `Pxxxx_context_ae_valid.json`의 `threshold_evaluation`
- `results/contextae_threshold_summary.csv`
- `results/contextae_threshold_summary.json`

UI의 모델 metrics 패널에도 threshold 방법별 Event-F1, Point-F1,
Precision, Recall, FA/day 비교표가 표시됩니다.
