# LSTM-AD × paperib

**하이퍼 · 학습 명령:** [docs/baseline-하이퍼파라미터.md](../docs/baseline-하이퍼파라미터.md) §2.  
**참고 논문:** Malhotra et al., ESANN 2015 — [docs/baseline-참고논문.md](../docs/baseline-참고논문.md).

LSTM next-step forecast; anomaly score = `−MSE`.  
Primary thr = train residual percentile. Features / split / `M688` via OmniAnomaly `prepare_arrays`.  
Thr from **train** only. Runner default = **valid only** (HP tuning); `--with_test` for final table.  
Labeling UI keeps valid|test Metrics controls.

## Observation (P0480 · primary · valid)

Train residual **percentile(1%)** thr는 valid에서 알람을 **남발**하기 쉽다.

- **Precision 낮음** (정상 residual도 fire → FP↑).
- **Recall은 P보다 높지만 낮은 편** (GT를 일부만 덮음).
- 예: point P≈0.15 / R≈0.43 / F1≈0.22 (설정·PLMN 한정).
- best-F1은 score 초극단 꼬리 thr → pred 1~소수; **비교용 아님**.

상세 서술: [docs/논문-스토리라인.md](../docs/논문-스토리라인.md) §4.3.

## Hyperparameters (improved defaults)

| Param | Default | Notes |
|-------|---------|-------|
| `--window` | **100** | lookback (was 32; aligned with OA/AT) |
| `--hidden` / `--layers` | **128** / 2 | was 64 / 2 |
| `--epochs` | **10** | aligned with OA/AT (was briefly 20) |
| `--batch_size` / `--lr` | 64 / 1e-3 | |
| `--anomaly_ratio` | 1.0 | train −MSE low percentile → thr |
| `--with_test` | off | also score/export test |

## Run

```bash
cd LSTM_AD
../.venv/bin/python run_plmn.py --plmn P0480
../.venv/bin/python run_plmn.py --plmn P0480 --anomaly_ratio 2.0
../.venv/bin/python run_plmn.py --plmn P0480 --run_name comb
../.venv/bin/python run_plmn.py --plmn P0480 --with_test
```

## Outputs

| | Path |
|--|------|
| Predictions | `data/ib_data/predictions/{PLMN}_lstm_ad[_{run}].json` |
| Model / metrics | `LSTM_AD/model/{PLMN}/{run}/lstm_ad/` |
