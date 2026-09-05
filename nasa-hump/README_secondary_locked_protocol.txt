NASA hump secondary locked evaluation protocol

Why "locked secondary evaluation" rather than "never-viewed untouched test":
The current full-grid post-processing code is capable of evaluating all 4,761 LES
mean-field points. Therefore a pristine "never viewed" historical claim cannot be
certified from the available files alone. This protocol instead adds a transparent,
outcome-independent secondary robustness check.

Frozen membership rule:
- Start only from split == train_pool_unused.
- Treat all columns belonging to prior train, validation, or original test partitions
  as prior-use columns.
- Retain every unused point whose structured i_index is at least 2 columns away from
  every prior-use column.
- No model prediction, error, or uncertainty quantity enters the selection rule.

Frozen result for the supplied manifest:
- Points: 336
- Structured i-columns: [41, 51, 58, 63, 70, 71, 72, 73, 103, 107, 114, 115, 122, 123, 156, 157]
- Original-index SHA256: 19dc4889f6ebc926b52191b5bf8505b47f034af59528ffd18f20ddf53d957c62
- x range: [0.701955, 1.203105]
- y range: [0.006511, 0.232386]

Recommended server placement:
  /hy-tmp/nasa-hump/hump_secondary_locked_evaluate.py
  /hy-tmp/nasa-hump/secondary_locked_protocol/secondary_locked_manifest.csv
  /hy-tmp/nasa-hump/secondary_locked_protocol/secondary_locked_protocol_summary.csv

Run exactly once on the frozen final checkpoints:
  cd /hy-tmp/nasa-hump
  python hump_secondary_locked_evaluate.py \
    --source-manifest ./hump_results_revision_r02/protocol/interior_velocity_split_manifest.csv \
    --frozen-manifest ./secondary_locked_protocol/secondary_locked_manifest.csv \
    --models-root ./hump_results_revision_r02/models \
    --output-dir ./hump_results_revision_r02/secondary_locked_evaluation \
    --mc-samples 50 \
    --dropout-rate 0.002 \
    --seed 424242

Do not retrain, change lambda_f, change dropout, or alter the frozen manifest after
seeing the secondary results.

The evaluator also reports MC50 predictive-mean error separately from the B-PINN
dropout-off prediction, so the two point-prediction conventions cannot be conflated.
