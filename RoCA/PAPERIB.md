# RoCA × paperib

Official contamination-robust TSAD ([ruiking04/RoCA](https://github.com/ruiking04/RoCA), MIT) adapted for IB comparison with RiAD.

- Network / loss / augmentations: `RoCA/official/` (upstream excerpts)
- Runner: Merlion-free; same valid-only export + VUS-PR metrics as RiAD
- Score: distance-to-center (`loss_n`, **higher = anomaly**)
- Thr: train top-percentile (`--anomaly_ratio`)

## Run (Operator 1 = P0480)

```bash
cd RoCA
../.venv/bin/python run_plmn.py --plmn P0480 --epochs 20
```

Export: `data/ib_data/predictions/P0480_roca.json`  
Checkpoint: `RoCA/model/P0480/paperib/roca.pt`

라벨링 UI에서 예측 소스 `RoCA · …` 로 RiAD와 전환 비교.

## Defaults

SWaT-style multivariate config: `window=32`, `time_step=16` (train), dense `time_step=1` scoring for VUS; `train_method=contaminated`, `nu=0.01`.
