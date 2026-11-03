# results/

실험 지표·스윕 요약 (가중치 `.pt` 제외). Codex / 원격에서도 동일 숫자를 보도록 git에 추적한다.

```text
results/
├── riad/{PLMN}/
│   ├── valid_metrics.json
│   ├── hp_random_results.csv
│   └── …
└── roca/{PLMN}/
    └── valid_metrics.json
```

예측 시계열 전체는 `data/ib_data/predictions/*.json`.
라벨은 `data/ib_data/labels/*_labels.json`.
