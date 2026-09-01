# data/

원본·라벨·역매핑을 둡니다.

```text
data/masked_YYYYMMDD.csv          # gitignore
data/plmn_mapping.txt             # gitignore
data/metric_mapping.txt           # gitignore
data/labels/{PLMN}_labels.json    # anomaly 라벨 (git 추적)
data/labels/plmn_rank.csv         # preprocess 산출 (gitignore)
```

`top100.txt`에 있는 사업자만 포함한다고 가정합니다. CSV 추가 후 `python labeling/preprocess.py` 실행.
