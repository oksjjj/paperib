# data/

통일 레이아웃 (데이터셋마다 `raw/` + `pickle/`):

```text
data/
├── ib_data/                         # paperib (IB)
│   ├── raw/                         # paper-only masked_*.csv (git) + local *_mapping.txt (ignored)
│   ├── cache/                       # 공통 PLMN 로딩 캐시 (git 제외)
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
python scripts/build_ui_cache.py
python ui/app.py
```

## 공개 벤치마크

```bash
# repo 루트에서
.venv/bin/python scripts/download_smd.py          # → data/SMD/raw/
.venv/bin/python scripts/preprocess_data.py --dataset SMD  # → data/SMD/pickle/
```
