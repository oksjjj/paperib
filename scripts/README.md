# scripts/

Repo 루트 공용 스크립트 (OmniAnomaly / AnomalyTransformer).

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

학습은 여전히 `OmniAnomaly/main.py` 에서 실행합니다.
