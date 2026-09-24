# OmniAnomaly × paperib

PyTorch OmniAnomaly (KDD'19 port) vendored for paperib PLMN timeseries.

## Setup

From the paperib repo root:

```bash
source .venv/bin/activate
pip install -r OmniAnomaly/requirements.txt
```

## Train & export UI predictions (P0480)

```bash
cd OmniAnomaly
../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 10 --stable_train
```

Default chronological split: **train 60% / valid 20% / test 20%**.

- Detection metrics + UI overlay use **valid only**
- **test is held out** (not scored) until you intentionally evaluate it
- Change ratios with `--train_ratio` / `--valid_ratio`

TensorBoard is **on by default** (train/valid loss, lr). Disable with `--no_tensorboard`.

### View training curves

```bash
cd OmniAnomaly
../.venv/bin/python view_tensorboard.py --plmn P0480
# → http://127.0.0.1:6006/
```

Logs live under `log/{PLMN}/{run}/tensorboard/`.

What it does:

1. Loads `P0480` via `labeling/tool.load_plmn`
2. Builds feature matrix (raw metrics, no rate overlays) for `paperib`
3. Chronological train/valid/test split; drops human-labeled anomaly rows from train
4. Trains OmniAnomaly, scores the **valid** split for UI
5. Sets a POT threshold from train scores
6. Writes model/result/log and `data/ib_data/predictions/{PLMN}_omnianomaly.json`

Re-score an existing checkpoint:

```bash
../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 0 \
  --restore_dir model/P0480/paperib
```

## Model runs (`--run_name`)

| `run_name` | Features | Standardization |
|------------|----------|-----------------|
| `paperib` (default) | All raw metrics | MinMax (train fit) |
| `comb` | COMB counters + S_RATE/A_RATE | MinMax (train fit) |
| `comb_share` | M971 + counter÷M971 | log1p(M971)+MinMax; shares MinMax |

**Why COMB runs exist and how `comb` differs from `comb_share`:** see **[COMB_RUNS.md](./COMB_RUNS.md)** (Korean).

Quick commands:

```bash
../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 10 --stable_train --run_name comb
../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 10 --stable_train --run_name comb_share
```

## Labeling UI

Open the labeling app and use the overlay radio:

- **사람+모델** — human (yellow) + OmniAnomaly (purple dashed)
- **사람만** / **모델만** / **숨김**

Use **모델 run** to switch `paperib` / `comb` / `comb_share` predictions.

Model predictions are read-only and never written into `data/ib_data/labels/*_labels.json`.

### Metric attribution (why this segment?)

Each predicted segment in `data/ib_data/predictions/{PLMN}_omnianomaly*.json` can include `top_metrics`:

- **contribution** = `train_median(log_prob) − segment_mean(log_prob)`
- Larger contribution ⇒ that metric reconstructed worse than usual

Re-export attribution without full retrain:

```bash
../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 0 \
  --restore_dir model/P0480/paperib
```
