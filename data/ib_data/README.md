# ib_data/ (`data/ib_data/`)

```text
data/ib_data/
├── raw/                         # masked_*.csv + local-only *_mapping.txt (gitignored)
├── pickle/                      # PLMN Omni/AT 학습 캐시
│   ├── {PLMN}_train.pkl …
│   ├── comb/
│   ├── comb_share/
│   └── paperib/
├── labels/{PLMN}_labels.json
└── predictions/{PLMN}_*.json
```

**보안:** `plmn_mapping.txt` / `metric_mapping.txt`(masked ↔ 원본명)는 **로컬 전용**이다.  
git·문서·이슈·주석에 대응표를 올리지 않는다.

```bash
python labeling/preprocess.py
python labeling/app.py
```

```bash
cd OmniAnomaly && ../.venv/bin/python run_plmn.py --plmn P0480 --run_name comb
# → data/ib_data/predictions/… , data/ib_data/pickle/comb/…
```
