#!/usr/bin/env python3
"""SITL ablation harness for Eagle Eye X — research-grade experiment runner.

Runs the ASPS tracking pipeline over a configuration matrix and collects
structured logs + metadata sidecars for the paper.

Matrix dimensions (extend the MATRIX list freely):
  - USE_ASPS: True (ASPS) vs False (pure class-priority baseline)
  - ASPS weights / hysteresis variations
  - LiDAR mock vs real (add USE_REAL_LIDAR + LIDAR_PORT to a cell)

Each trial:
  1. writes experiments/results/<cell>/config.yaml
  2. launches src/ASPS_LIDAR/ASPS_LIDAR_TR.py --config that file (subprocess)
  3. the script self-lands after RUN_DURATION_S seconds
  4. writes run_metadata.yaml (config, git hash, timestamp)
Finally runs tools/analyze_flight.py across all results for a comparison table.

Requirements: ArduPilot SITL reachable at udpin://127.0.0.1:14550, and a camera
(CAMERA_INDEX) available to the tracking script. Mock LiDAR needs no hardware.

Usage:
    python experiments/ablation/run_ablation.py --dry-run         # write configs, no launch
    python experiments/ablation/run_ablation.py --duration 30
    python experiments/ablation/run_ablation.py --repeats 3 --cells asps_default,baseline_class
"""
import argparse
import datetime
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "results")

# --- Ablation matrix -------------------------------------------------------
MATRIX = [
    {
        "name": "asps_default",
        "USE_ASPS": True,
        "ALPHA": 0.60, "BETA": 0.20, "GAMMA": 0.15, "DELTA": 0.05,
        "SWITCH_THRESHOLD": 0.15,
    },
    {
        "name": "baseline_class",
        "USE_ASPS": False,
    },
    {
        "name": "asps_balanced",
        "USE_ASPS": True,
        "ALPHA": 0.25, "BETA": 0.25, "GAMMA": 0.25, "DELTA": 0.25,
        "SWITCH_THRESHOLD": 0.10,
    },
    {
        "name": "asps_velocity_boost",
        "USE_ASPS": True,
        "ALPHA": 0.45, "BETA": 0.20, "GAMMA": 0.15, "DELTA": 0.20,
        "SWITCH_THRESHOLD": 0.15,
    },
]


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Run the ASPS SITL ablation matrix")
    ap.add_argument("--repeats", type=int, default=1, help="trials per cell (statistical power)")
    ap.add_argument("--duration", type=int, default=30, help="seconds per trial (RUN_DURATION_S)")
    ap.add_argument("--dry-run", action="store_true", help="write configs + print commands only")
    ap.add_argument("--cells", default=None, help="comma-separated subset of matrix cell names")
    args = ap.parse_args()

    cells = MATRIX
    if args.cells:
        wanted = {c.strip() for c in args.cells.split(",")}
        cells = [c for c in MATRIX if c["name"] in wanted]
    if not cells:
        raise SystemExit("No matching cells in MATRIX")

    for cell in cells:
        for rep in range(args.repeats):
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(RESULTS, f"{cell['name']}_r{rep}_{stamp}")
            os.makedirs(run_dir, exist_ok=True)

            overrides = {k: v for k, v in cell.items() if k != "name"}
            overrides.update({
                "RUN_DURATION_S": args.duration,
                "VIDEO_FOLDER": run_dir,   # log CSV lands directly in the trial dir
                "RECORD_VIDEO": False,
            })

            cfg_path = os.path.join(run_dir, "config.yaml")
            with open(cfg_path, "w") as fh:
                yaml.safe_dump(overrides, fh, sort_keys=False)

            meta = {
                "cell": cell["name"],
                "repeat": rep,
                "timestamp": stamp,
                "duration_s": args.duration,
                "git_rev": git_rev(),
                "config": overrides,
            }
            with open(os.path.join(run_dir, "run_metadata.yaml"), "w") as fh:
                yaml.safe_dump(meta, fh, sort_keys=False)

            cmd = [sys.executable, "src/ASPS_LIDAR/ASPS_LIDAR_TR.py", "--config", cfg_path]
            print(f"[{stamp}] {cell['name']} rep {rep}: {' '.join(cmd)}")
            if args.dry_run:
                print(f"          -> {run_dir} (dry-run)")
                continue

            proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
            try:
                proc.wait(timeout=args.duration + 60)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"          exit={proc.returncode} -> {run_dir}")

    if not args.dry_run:
        print("\nRunning comparison analysis across all trials...")
        subprocess.run(
            [sys.executable, "tools/analyze_flight.py", RESULTS, "-r",
             "--summary-out", os.path.join(RESULTS, "comparison_summary.csv")],
            cwd=REPO_ROOT, check=False,
        )
        print(f"\nComparison table: {os.path.join(RESULTS, 'comparison_summary.csv')}")


if __name__ == "__main__":
    main()
