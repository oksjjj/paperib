# scripts/

Repo 루트 공용 데이터 전처리·평가·시각화 스크립트.

```bash
# from paperib root
.venv/bin/python scripts/download_smd.py
.venv/bin/python scripts/preprocess_data.py --dataset SMD

.venv/bin/python scripts/download_smap_msl.py
.venv/bin/python scripts/preprocess_data.py --dataset SMAP
.venv/bin/python scripts/preprocess_data.py --dataset MSL

.venv/bin/python scripts/eval_from_scores.py --dataset machine-1-1
.venv/bin/python scripts/viz_gt_anomalies.py --dataset machine-1-1 --gt_only
```



## VUS 재평가

저장된 validation score JSON을 사용해 모델 재학습 없이 VUS 지표를 계산합니다.

```bash
.venv/bin/python scripts/eval_vus.py --all
```

모델을 제한하려면:

```bash
.venv/bin/python scripts/eval_vus.py --all --models context_ae hour_ae roca
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
.venv/bin/python scripts/eval_contextae_thresholds.py --all
```

한 사업자만:

```bash
.venv/bin/python scripts/eval_contextae_thresholds.py --plmn P0480
```

기본 후보를 직접 지정하려면:

```bash
.venv/bin/python scripts/eval_contextae_thresholds.py \
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
