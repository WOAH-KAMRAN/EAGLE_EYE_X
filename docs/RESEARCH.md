# Research Methodology & Reproducibility

This document defines the metrics and experimental protocol so the paper's results
are reproducible by any team member. All artifacts for a given experiment must be
recreatable from this repo + the private data/models.

## 1. Metrics (from flight logs)

All tracking-error metrics are computed **over `OFFBOARD` rows only**
(`tools/analyze_flight.py` enforces this).

| Metric | Definition |
|---|---|
| Yaw error RMSE / MAE | `sqrt(mean(yaw_err²))`, `mean(abs(yaw_err))` in px |
| Pitch error RMSE / MAE | same for `pitch_err` |
| Error P95 | 95th percentile of `abs(error)` — tail-behavior indicator |
| Altitude error | RMSE/MAE of `Alt_Err` |
| Target switches | count of `Class` transitions in the log |
| Lock retention | time span between first and last tracked target while OFFBOARD |
| Centered fraction | fraction of OFFBOARD rows with `Target_Centered == 1` |
| Distance-following error | RMSE/MAE of `KF_Dist_m − TARGET_DISTANCE_M` |
| ASPS components | mean of `BaseP, ConfP, DistP, VelP, FinalP` (per-cell ablation insight) |

### Generate metrics + figures for any log

```bash
python tools/analyze_flight.py flights/flight_data_20250916_103253/flight_log_20250916_103253.csv --figdir results/figures
python tools/analyze_flight.py flights/ -r --summary-out results/figures/summary.csv   # all logs
```

Figures produced per log: error curves, target/mode switching, ASPS component
breakdown, distance-following plot.

## 2. Ablation protocol (ASPS)

The core ablation isolates the contribution of the Adaptive Semantic Priority
Selection against the pure class-priority baseline.

Cells (in `experiments/ablation/run_ablation.py`):

| Cell | `USE_ASPS` | Weights `(α,β,γ,δ)` | Purpose |
|---|---|---|---|
| `asps_default` | True | (0.60, 0.20, 0.15, 0.05) | Production configuration |
| `baseline_class` | False | — | Pure `max(CLASS_PRIORITIES)` baseline |
| `asps_balanced` | True | (0.25, 0.25, 0.25, 0.25) | Equal-weight sensitivity |
| `asps_velocity_boost` | True | (0.45, 0.20, 0.15, 0.20) | Velocity-term sensitivity |

Run protocol:

```bash
# check the matrix without launching
python experiments/ablation/run_ablation.py --dry-run

# run all cells, 30 s each, 3 repeats for statistics (SITL + camera required)
python experiments/ablation/run_ablation.py --duration 30 --repeats 3
```

Each trial writes:

```
experiments/results/<cell>_r<rep>_<timestamp>/
├── config.yaml            # exact runtime config (reproducibility)
├── run_metadata.yaml      # cell, git_rev, timestamp, duration
└── log_<timestamp>.csv    # full flight log
```

The harness then auto-runs `tools/analyze_flight.py` across all trials and writes
`experiments/results/comparison_summary.csv` — the raw input for paper tables.

### Reporting rules

- Report **mean ± std** across repeats, never a single run.
- State `git_rev`, SITL version, camera, and environment in the methodology.
- When a new script/log-schema change lands, re-run the matrix — do not mix logs
  from different schemas in one table.

## 3. Figure recipes (paper)

1. **ASPS component evolution** — `tools/analyze_flight.py` `*_asps_components.png`
   (BaseP/ConfP/DistP/VelP + FinalP vs time). Use for a representative tracking scene.
2. **Tracking-error CDF/curves** — `*_error_curves.png`; compute RMSE/MAE table from
   the summary CSV.
3. **Ablation bar chart** — read `comparison_summary.csv`; plot target-switch count,
   lock retention, and yaw-error RMSE per cell with error bars.

## 4. Reproducibility checklist before submission

- [ ] `requirements.txt` + `requirements-jetson.txt` current
- [ ] Ablation results regenerated on the merged repo state (`git_rev` recorded)
- [ ] Model card present (`docs/MODEL_TESTING.md`) with mAP/P/R
- [ ] All figures traceable to a log + script version
- [ ] Private data/models NOT committed; paths injected via `$EAGLE_DATASET`
