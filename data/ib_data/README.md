# ib_data/ (`data/ib_data/`)

```text
data/ib_data/
├── raw/masked_*.csv          # paper 10 PLMN only · de-identified · git 추적
├── raw/*_mapping.txt         # 로컬 전용 · git 금지 (역매핑)
├── _local_raw_full/          # 전체 PLMN raw (로컬 전용 · git 금지)
├── cache/{PLMN}.pkl          # 공통 PLMN 로딩 캐시 · 로컬 전용
├── labels/{PLMN}_labels.json
└── predictions/{PLMN}_*.json
```

**보안:** `plmn_mapping.txt` / `metric_mapping.txt`(masked ↔ 원본명)는 **로컬 전용**.  
git·문서·이슈·주석에 대응표를 올리지 않는다.

## paper raw 재생성

전체 dump가 `_local_raw_full/`에 있을 때:

```bash
.venv/bin/python scripts/export_paper_raw.py
```

라벨링 캐시:

```bash
python scripts/build_ui_cache.py
# 기본값은 top100.txt 대상이며 --plmn / --top / --all 옵션도 사용할 수 있음
```
