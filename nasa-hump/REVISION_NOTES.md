
# NASA Hump Revision v1.2

## Purpose

This package extends v1.1 with an explicit boundary-treatment protocol for the
released NASA hump LES reconstruction window.

### Implemented now

1. **Steady 2-D architecture**
   - `(x, y) -> (psi, p)`
   - `u = d psi / d y`
   - `v = - d psi / d x`
   - 10x100 psi-p backbone: **91,402 trainable parameters**

2. **Strict-interior train / validation / held-out test**
   - All four edges of the structured LES window are removed before splitting.
   - The 2% setting still selects **95 strict-interior velocity observations**
     because the sparse ratio is expressed relative to the complete 4,761-point
     LES table.
   - Lower-ratio training subsets remain nested under the same seed.

3. **Physical hump wall**
   - Bottom structured-grid row: 207 points.
   - Training target: `u = 0`, `v = 0`.
   - Loss term: `wall_loss`.

4. **Truncated subdomain interfaces**
   - Left edge excluding corners: 21 points.
   - Right edge excluding corners: 21 points.
   - Top edge: 207 points.
   - Combined unique subdomain-boundary observations: **249 points**.
   - Their LES `u,v` values enter `subbc_loss` by default.
   - These are explicitly called *truncated subdomain boundary observations*,
     NOT the physical inlet/outlet/upper wall of the complete NASA benchmark.

5. **Geometry-aware collocation**
   - Formal default: `--collocation-mode fluid_domain`.
   - Collocation points are above the hump wall and inside the released local LES
     reconstruction window.
   - Legacy rectangular sampling remains available only for audit/reproduction.

6. **Pressure gauge**
   - `Cp = 2 p` by default.
   - Held-out and full-grid validation save both raw and gauge-aligned Cp metrics.

7. **Traceable loss logging**
   Every epoch records separately:
   - `uv_loss`
   - `equation_loss`
   - `cp_loss`
   - `wall_loss`
   - `subbc_loss`
   - weighted versions of all five
   - total loss
   - parameter count
   - train/validation/test/boundary point counts

## Important scientific bookkeeping

The `--supervised-ratio 0.02` setting means **2% sparse strict-interior velocity
observations relative to the full 4,761-point LES table**. The 207 physical-wall
constraints and, when enabled, the 249 subdomain-boundary LES observations are
reported separately and must NOT be hidden inside a claim of "only 2% total data."

For the final manuscript, report:
- sparse interior data ratio;
- physical wall constraint count;
- subdomain-boundary observation count;
- Cp observation count;
separately.

The command-line flags `--disable-wall-bc` and `--disable-subbc` are provided for
later ablation/control experiments.

## STEP 1 — overwrite only the Python revision files

Back up v1.1 first (already done), then unzip v1.2 over the current working folder.

## STEP 2 — protocol-only check (DO THIS NEXT)

From `/hy-tmp/nasa-hump`:

```bash
python hump_train.py \
  --data-dir . \
  --results-root ./results_2pct_v12 \
  --supervised-ratio 0.02 \
  --n-equation-points 50000 \
  --collocation-mode fluid_domain \
  --split-mode spatial_blocks \
  --seed 2025 \
  --protocol-only
```

Inspect:

```bash
cat results_2pct_v12/protocol/protocol_summary.csv
ls -lh results_2pct_v12/protocol/
```

Expected structured-grid counts if the NASA file contains all 4,761 finite points:

```text
full_les_points                    4761
strict_interior_points             4305
hump_wall_points                    207
left_subdomain_points                21
right_subdomain_points               21
top_subdomain_points                207
combined_subdomain_boundary_points  249
interior_train_points                95
```

The exact validation/test counts are determined by the fixed spatial-block split.

New protocol files include:

```text
interior_velocity_split_manifest.csv
hump_wall_points.csv
left_subdomain_boundary.csv
right_subdomain_boundary.csv
top_subdomain_boundary.csv
subdomain_boundary_points.csv
cp_split_manifest.csv
valid_fluid_collocation_points.csv
legacy_random_box_diagnostic_all_points.csv
legacy_outside_local_reference_window_points.csv
protocol_summary.csv
protocol_boundary_map.png
legacy_sampler_audit.png
```

## STEP 3 — 5-epoch boundary smoke test

Only after STEP 2 looks correct:

```bash
python hump_train.py \
  --data-dir . \
  --results-root ./results_smoke_v12 \
  --method standard_pinn \
  --epochs 5 \
  --supervised-ratio 0.02 \
  --n-equation-points 2000 \
  --collocation-mode fluid_domain \
  --split-mode spatial_blocks \
  --seed 2025
```

Look for finite values for all five losses:

```text
uv=
eq=
cp=
wall=
subbc=
```

and verify:

```text
Parameters: 91402
```

## DO NOT start formal 2000-epoch runs yet

The next task after the v1.2 smoke test is the Reviewer #21 reference-field
residual audit. We should evaluate the exact simplified momentum residual on the
original LES reference field before choosing/finalizing the physics-loss weight.

Only after that audit should the formal 1--5% and baseline runs begin.
