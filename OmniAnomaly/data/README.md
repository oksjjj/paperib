# Data

이 디렉터리는 git에 추적하지 않습니다 (대용량). 클론 후 아래처럼 준비하세요.

```
data/
├── SMAP/                         # 학습용: SMAP_{train,test,test_label}.pkl
├── MSL/                          # 학습용: MSL_{train,test,test_label}.pkl
├── SMD/                          # 학습용: machine-*_{train,test,test_label}.pkl
└── raw/
    ├── nasa/                     # SMAP/MSL 원본 (train/, test/, labeled_anomalies.csv)
    └── ServerMachineDataset/     # SMD 원본 (train/, test/, test_label/)
```

## 준비

```bash
# SMD
python scripts/download_smd.py
python scripts/preprocess_data.py --dataset SMD

# SMAP / MSL
python scripts/download_smap_msl.py
python scripts/preprocess_data.py --dataset SMAP
python scripts/preprocess_data.py --dataset MSL
```
