# RiAD × paperib

Contamination-robust multivariate TSAD backbone (Chen et al., *Information Sciences* 2024).  
논문 스토리: [docs/논문-스토리라인.md](../docs/논문-스토리라인.md).

이 폴더는 원저자 미공개 코드를 복제한 것이 아니라, RiAD의 핵심 역할  
(불확실성 재구성 · adaptive loss · anomaly injection · awareness memory)을  
IB 파이프라인에 맞게 구현한 **practical backbone**이다.

## 학습 · valid 평가 (Operator 1 = P0480)

```bash
cd RiAD
../.venv/bin/python run_plmn.py --plmn P0480 --epochs 10
```

## 하이퍼파라미터 스윕 (valid only)

```bash
cd RiAD
# 1) thr만 빠르게 (1회 학습 후 anomaly_ratio 스윕) — 추천 시작점
../.venv/bin/python hp_search.py --mode thr --plmn P0480 --export_best

# 2) 기본 grid (ar × lr × hidden × inject_p …)
../.venv/bin/python hp_search.py --mode grid --plmn P0480 --export_best

# 3) random
../.venv/bin/python hp_search.py --mode random --n_trials 20 --plmn P0480
```

결과: `RiAD/model/{PLMN}/hp_search/{mode}_{timestamp}/`
- `results.csv` / `results.json` / `best.json`
- 정렬: **VUS-PR ↓ → Event-F1 ↓ → FA/day ↑**
- `--export_best`: 최적 설정을 `data/ib_data/predictions/{PLMN}_riad.json`으로 export

- train 60% / valid 20% / test 20% (chronological)
- feature: 전체 raw counter (`paperib`)
- primary thr: train score percentile (`--anomaly_ratio`, default 1.0)
- score: reconstruction MSE (**높을수록 이상**; `score > thr`)
- export: `data/ib_data/predictions/{PLMN}_riad.json` (**valid only**)
- checkpoint: `RiAD/model/{PLMN}/paperib/riad.pt`

라벨링 UI에서 예측 소스 `RiAD · paperib …` 로 오버레이된다.

## 평가 지표 (논문 스토리)

| 역할 | 지표 |
|------|------|
| **Primary** | VUS-PR |
| Event | Event-F1 |
| Additional | PR-AUC, Precision, Recall, F1 |
| Operational | False Alarms / day, Detection Delay |

**제외(주요):** Point-Adjusted F1, Accuracy. AUROC는 보조만.

구현: `RiAD/metrics.py` (`evaluate_riad_valid`).
