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
