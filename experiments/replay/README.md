# Offline ASPS replay

Compares target-selection policies (ASPS variants vs simple baselines) on recorded
video, with no drone, SITL or LiDAR. Produces the numbers for the IRIS'27 abstract
and the paper's selection-ablation table.

```bash
source VENV/uav/bin/activate

# 1. detect + track once, score all policies (use best.engine on the Orin)
python experiments/replay/replay_asps.py --source clip.mp4 --model models/best.pt

# 2. hand-label which vehicle SHOULD be followed (every 5th frame)
python experiments/replay/label_gt.py --source clip.mp4 \
    --tracks experiments/replay/results/clip_<stamp>/tracks.json --out clip_gt.csv

# 3. re-score with ground truth (no re-detection)
python experiments/replay/replay_asps.py \
    --cache experiments/replay/results/clip_<stamp>/tracks.json --gt clip_gt.csv
```

Per run (`experiments/replay/results/<name>_<stamp>/`, gitignored): `tracks.json`,
`sel_<policy>.csv`, `summary.csv`, `run_metadata.yaml` (git rev, model, inference ms/FPS).

| Metric | Meaning |
|---|---|
| `selection_accuracy` / `wrong_target_rate` | selected box IoU ≥ 0.5 with the labelled correct target (needs `--gt`) |
| `switches_per_min`, `flicker_switches` | target changes; flicker = switch away and back within 1 s |
| `acq_delay_mean_s`, `acq_missed` | time to lock a newly-appeared higher-priority vehicle |
| `kf_jitter_reduction_pct` | frame-to-frame centre jitter, raw vs Kalman (same KF as the flight script) |
| `coverage` | frames with detections where a target was selected |
| `infer_ms_mean`, `infer_fps` (metadata) | detector speed on this machine; run on the Orin for paper numbers |

Notes for the paper: tracking is a dependency-free IoU/Hungarian tracker; distance is a
box-height proxy (`--focal-px`, `--obj-h-m`); ASPS state is keyed by track id (the flight
script keys it by class id). `asps_hold` / `class_priority_hold` add a 10-frame dropout
hold; compare them to see what hold alone contributes.
