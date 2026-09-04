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
2. Builds feature matrix (raw metrics, no rate overlays)
3. Chronological train/test split (default 70/30); drops human-labeled anomaly rows from train
4. Trains OmniAnomaly, scores the **full** series
5. Sets a POT threshold from train scores
6. Writes:
   - `OmniAnomaly/model/P0480/paperib/`
   - `OmniAnomaly/result/P0480/paperib/`
   - `data/predictions/P0480_omnianomaly.json` ← labeling UI overlay

Re-score an existing checkpoint:

```bash
../.venv/bin/python run_plmn.py --plmn P0480 --max_epoch 0 \
  --restore_dir model/P0480/paperib
```

## Labeling UI

Open the labeling app and use the overlay radio:

- **사람+모델** — human (yellow) + OmniAnomaly (purple dashed)
- **사람만** / **모델만** / **숨김**

Model predictions are read-only and never written into `data/labels/*_labels.json`.
