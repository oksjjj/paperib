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

