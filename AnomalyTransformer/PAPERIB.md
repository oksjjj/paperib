# Anomaly Transformer × paperib

Vendored from the official [Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer)
(ICLR 2022 Spotlight) and wired like OmniAnomaly for PLMN runs.

## Setup

```bash
source .venv/bin/activate
# same stack as OmniAnomaly (torch, sklearn, …)
pip install -r OmniAnomaly/requirements.txt
```

## Train & export UI predictions

```bash
cd AnomalyTransformer
../.venv/bin/python run_plmn.py --plmn P0480 --num_epochs 10
../.venv/bin/python run_plmn.py --plmn P0480 --num_epochs 10 --run_name comb
../.venv/bin/python run_plmn.py --plmn P0480 --num_epochs 10 --run_name comb_share
```

Chronological split matches OmniAnomaly: **train 60% / valid 20% / test 20%**.
Detection + UI overlay use **valid only**; test is held out.

### Outputs

| `run_name` | Prediction JSON |
|------------|-----------------|
| `paperib` | `data/ib_data/predictions/{PLMN}_anomalytransformer.json` |
| `comb` | `data/ib_data/predictions/{PLMN}_anomalytransformer_comb.json` |
| `comb_share` | `data/ib_data/predictions/{PLMN}_anomalytransformer_comb_share.json` |

Checkpoints: `AnomalyTransformer/model/{PLMN}/{run_name}/`.

Re-score an existing checkpoint:

```bash
../.venv/bin/python run_plmn.py --plmn P0480 --num_epochs 0 --run_name comb
```

## Feature views (`--run_name`)

Same three views as OmniAnomaly (prep reused from `OmniAnomaly/run_plmn.py`):

| `run_name` | Features | Standardization |
|------------|----------|-----------------|
| `paperib` | All raw metrics except `EXCLUDED_TRAIN_METRICS` (default drop: `M688`) | MinMax (train fit) |
| `comb` | COMB raw + S_RATE/A_RATE | MinMax (train fit) |
| `comb_share` | M971 + counter÷M971 | log1p(M971)+MinMax; share MinMax |

See [OmniAnomaly/COMB_RUNS.md](../OmniAnomaly/COMB_RUNS.md).

## Scoring note

Official energy is **higher = more anomalous**. For the labeling UI we store
``score = log10(energy_threshold) - log10(energy)`` so **threshold = 0**,
scores are O(1), and **anomaly ⇔ score &lt; 0**.

Energy threshold itself comes from the train-energy percentile
`100 - anormly_ratio` (default `anormly_ratio=1`).

Attribution uses per-dimension reconstruction MSE (not association maps).

## Labeling UI

Restart the labeling app and pick a **모델 run** whose label starts with `[AT]`.
