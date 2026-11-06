# MSRS: Multi-Scale Robust Sparse-aware TSAD

Research prototype for the paperib thesis.

## Research constraint

The backbone must train on an unlabeled, potentially contaminated train split.

This implementation therefore does not read any anomaly label file until all of the following are complete:

1. model training,
2. train residual calibration,
3. train-score threshold selection,
4. validation/test score generation.

Human labels are used only for evaluation after those steps.

## Method

For each temporal scale (default: 8 and 32):

1. GRU next-step prediction on all raw counters,
2. Huber warm-up training,
3. MAD-based soft down-weighting of unusually high-loss train windows,
4. feature-wise robust residual normalization using train median/MAD,
5. log-compressed top-k feature residual mean.

The final score is the maximum across temporal scales so that an anomaly visible at either short or long context is preserved.

No SKT-specific metric rule or anomaly label is used in the model.

## Default P0480 validation run

From repository root:

    cd MSRS
    ../.venv/bin/python run_plmn.py --plmn P0480

Outputs:

- data/ib_data/predictions/P0480_msrs_valid.json
- MSRS/model/P0480/paperib/msrs.pt

## Held-out test

Do not use the test split while selecting architecture or hyperparameters.

After validation choices are frozen:

    cd MSRS
    ../.venv/bin/python run_plmn.py --plmn P0480 --eval_test

The test threshold remains train-derived.
The VUS tolerance used on test is fixed from validation and is not selected from test labels.

## Useful ablations

Sparse-aware scoring:

    ../.venv/bin/python run_plmn.py --plmn P0480 --topk_ratio 0.10
    ../.venv/bin/python run_plmn.py --plmn P0480 --topk_ratio 1.00

Multi-scale vs single-scale:

    ../.venv/bin/python run_plmn.py --plmn P0480 --scales 8
    ../.venv/bin/python run_plmn.py --plmn P0480 --scales 32
    ../.venv/bin/python run_plmn.py --plmn P0480 --scales 8,32

Contamination-aware weighting ablation can be approximated by setting a very large robust tau:

    ../.venv/bin/python run_plmn.py --plmn P0480 --robust_tau 1000000

Primary diagnostics are VUS-PR, Event-F1, Event Recall, FA/day, and per-GT-event max-score percentile.
