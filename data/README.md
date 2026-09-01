# data/

마스킹 원본 CSV와 역매핑 파일을 둡니다. **git에 올리지 않습니다.**

```text
data/masked_YYYYMMDD.csv
data/plmn_mapping.txt      # masked_plmn ↔ original_plmn (TSV)
data/metric_mapping.txt    # masked_metric ↔ original_metric (TSV)
```

`top100.txt`에 있는 사업자만 포함한다고 가정합니다. CSV 추가 후 `python labeling/preprocess.py` 실행.
