# HourIF: 24-hour contextual iterative Isolation Forest

This is a point-based detector using 24 time-of-day models.

## Training flow

For each local hour 00 to 23:

1. Select unlabeled TRAIN points for that hour.
2. Fit RobustScaler + stage-1 Isolation Forest.
3. Remove the stage-1 anomaly candidates (default 5%).
4. Refit RobustScaler + final Isolation Forest on retained points.
5. Compute final clean-train anomaly-score mean/std.
6. Calibrate each point as z = (raw IF anomaly score - hour mean) / hour std.
7. Predict anomaly when z > threshold sigma (default 3.0).

Human labels are not loaded until all 24 models and train/valid/test scores are fixed.

## Important sklearn detail

IsolationForest contamination mainly sets its internal decision cutoff. The actual
representation change here comes from removing stage-1 candidates and refitting
the final forest on the retained data. The final 0.01% contamination setting is
kept to match the earlier implementation, while paperib uses a separate
clean-train mean-plus-sigma threshold.

## Validation run

From repository root:

    cd HourIF
    ../.venv/bin/python run_plmn.py --plmn P0480

Outputs:

    data/ib_data/predictions/P0480_hour_if_valid.json
    HourIF/model/P0480/paperib/hour_if.pkl

The model pkl is ignored by git.

## Held-out test

After validation settings are frozen:

    ../.venv/bin/python run_plmn.py --plmn P0480 --eval_test

## Defaults

    --timezone Asia/Seoul
    --stage1_remove_pct 5.0
    --stage2_contamination_pct 0.01
    --threshold_sigma 3.0
    --n_estimators 300

The time zone can be changed without changing the algorithm:

    ../.venv/bin/python run_plmn.py --plmn P0480 --timezone UTC
