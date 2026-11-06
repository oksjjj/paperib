# HourAE: 24-hour contextual iterative AutoEncoder

This reproduces the user's earlier point-based strategy using 24 AutoEncoders.

## Pipeline

For each local hour 00..23:

1. Train a point-wise AutoEncoder on all unlabeled TRAIN points in that hour.
2. Compute point reconstruction MSE.
3. Remove the top 5% reconstruction-error points.
4. Train a fresh AutoEncoder on the retained points.
5. Compute the final clean-train MSE distribution.
6. Score each point using the AutoEncoder for its own hour.
7. Convert raw MSE to z = (MSE - hour mean) / hour std.
8. Predict anomaly when z > threshold sigma (default 3.0).

No anomaly label is used for model training, filtering, calibration, or score generation.

## About final contamination 0.01%

A neural AutoEncoder itself has no IsolationForest-like contamination parameter.

Therefore --final_contamination_pct 0.01 is recorded as the reference upper-tail
quantile of the final clean-train reconstruction errors for each hour. It does
not alter AutoEncoder training weights.

The actual decision threshold is kept separate and uses the clean-train
mean-plus-sigma rule.

## Default architecture

Input F -> 32 -> 8 -> 32 -> F

- ReLU
- MSE reconstruction loss
- Adam optimizer
- RobustScaler fitted independently for each hour and training stage
- stage-1 epochs: 50
- final epochs: 100

## Validation

From repository root:

    cd HourAE
    ../.venv/bin/python run_plmn.py --plmn P0480

Output:

    data/ib_data/predictions/P0480_hour_ae_valid.json

Checkpoint:

    HourAE/model/P0480/paperib/hour_ae.pt

The checkpoint is ignored by git through the existing *.pt rule.

## Main options

    --timezone Asia/Seoul
    --stage1_remove_pct 5
    --final_contamination_pct 0.01
    --threshold_sigma 3
    --hidden_dim 32
    --latent_dim 8
    --stage1_epochs 50
    --final_epochs 100
    --batch_size 128
    --lr 0.001

## Held-out test

After method and threshold choices are frozen:

    ../.venv/bin/python run_plmn.py --plmn P0480 --eval_test
