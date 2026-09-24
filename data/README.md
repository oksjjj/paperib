# data/

통일 레이아웃 (데이터셋마다 `raw/` + `pickle/`):

```text
data/
├── ib_data/                         # paperib (IB)
│   ├── raw/                         # masked_*.csv (+ local-only *_mapping.txt, gitignored)
│   ├── pickle/                      # PLMN 학습 캐시 (train/valid/test pkl)
│   ├── labels/
│   └── predictions/
├── SMD/
│   ├── raw/                         # train/ test/ test_label/ …
│   └── pickle/                      # machine-*_{train,test,test_label}.pkl
├── SMAP/
│   ├── raw/
│   └── pickle/
└── MSL/
    ├── raw/
    └── pickle/
```

## paperib

```bash
# CSV는 data/ib_data/raw/ 에 둠
python labeling/preprocess.py
python labeling/app.py
```

## 공개 벤치마크

```bash
# repo 루트에서
.venv/bin/python scripts/download_smd.py          # → data/SMD/raw/
.venv/bin/python scripts/preprocess_data.py --dataset SMD  # → data/SMD/pickle/
```
