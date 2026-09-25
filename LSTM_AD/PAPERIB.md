# LSTM-AD × paperib

Experiment forecasting 1-pick from [docs/논문-스토리라인.md](../docs/논문-스토리라인.md).

LSTM next-step forecast; anomaly score = `−MSE`.  
Primary thr = train residual percentile. Features / split / `M688` via OmniAnomaly `prepare_arrays`.

## Run

```bash
cd LSTM_AD
../.venv/bin/python run_plmn.py --plmn P0480 --epochs 10
../.venv/bin/python run_plmn.py --plmn P0480 --epochs 10 --run_name comb
```

## Outputs

| | Path |
|--|------|
| Predictions | `data/ib_data/predictions/{PLMN}_lstm_ad.json` |
| Checkpoint | `LSTM_AD/model/{PLMN}/{run}/lstm_ad/model.pt` |
