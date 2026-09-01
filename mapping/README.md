# mapping/

로컬 라벨링·분석용 PLMN/지표 역매핑 파일을 둡니다. **git에 올리지 않습니다.**

| 파일 | 형식 |
|------|------|
| `plmn_mapping.txt` | TSV · `masked_plmn`, `original_plmn` |
| `metric_mapping.txt` | TSV · `masked_metric`, `original_metric` |

파일이 없어도 앱은 동작합니다(마스킹 ID만 표시). `labeling/tool.py`가 이 경로를 읽습니다.
