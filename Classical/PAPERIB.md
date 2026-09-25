# Classical × paperib (IQR + EWMA)

Experiment classical 1-pick from [docs/논문-스토리라인.md](../docs/논문-스토리라인.md).

Per-feature **IQR** fences + **EWMA** residual spikes, fused with **OR**.  
UI score = `−(# fires)` (lower = more anomalous). `M688` excluded via shared prep.

## Run

```bash
cd Classical
../.venv/bin/python run_plmn.py --plmn P0480
../.venv/bin/python run_plmn.py --plmn P0480 --run_name comb
```

## Outputs

| | Path |
|--|------|
| Predictions | `data/ib_data/predictions/{PLMN}_classical.json` |
| Metrics | `Classical/model/{PLMN}/{run}/classical/` |
