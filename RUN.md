# RUN.md — how to run the system

Quick start for teammates. Longer versions: `docs/SETUP.md` (PC/Jetson),
`docs/HARDWARE.md` (wiring), `docs/MODEL_TESTING.md`, `docs/RESEARCH.md`.

> **Run everything from the repo root.** All model/video/log paths are relative to
> your current directory (`models/best.pt`, `flights/`, ...).

---

## 0. One-time setup

```bash
git clone git@github.com:WOAH-KAMRAN/EAGLE_EYE_X.git
cd EAGLE_EYE_X
make setup      # creates VENV/uav, installs requirements.txt
make models     # fetches dev weights into models/
```

- `models/best.pt` (custom 4-class model) is **private** — request it from a
  maintainer and drop it into `models/`.
- **Important:** after the history rewrite, re-clone — do not `git pull` a pre-rewrite
  checkout. If you still have an old clone: fetch + reset:

```bash
git fetch origin && git reset --hard origin/main
```

---

## 1. Simulation (SITL) — no hardware needed

Terminal A — start the simulator (or use an already-running SITL):

```bash
cd Tools/ardupilot/build/sitl/bin   # your ArduPilot SITL build
./arducopter --home 47.6,19.1,584,0
# accepts MAVLink on udp://127.0.0.1:14550 (SYSID 1)
```

Terminal B — run the tracking node:

```bash
source VENV/uav/bin/activate
make sitl
# python src/ASPS_LIDAR/ASPS_LIDAR_TR.py --config configs/sitl.example.yaml
```

On startup the script prints its **resolved settings** (model, connection, ASPS
weights, mock LiDAR, ...). Add `--verbose` to dump every resolved config key.

- SITL config: mock LiDAR, `models/yolov8n.pt`, camera off.
- **Engage:** switch the vehicle to `OFFBOARD`/`GUIDED` in QGC (or MAVProxy
  `mode guided`). While OFFBOARD the controller issues yaw/pitch rate commands
  that steer the detected target into the frame center.
- **Abort (always works):** `Ctrl+C` or press `Q` in the window — triggers an
  emergency LAND.

---

## 2. Real hardware (Jetson + Pixhawk + camera + LiDAR)

```bash
# 1. copy the hardware config to your private override and edit it
cp configs/hardware.yaml configs/local/hardware.yaml
$EDITOR configs/local/hardware.yaml    # serial port, LiDAR topic, camera index

# 2. run from the repo root (configs/local/*.yaml auto-layered on top of configs/*.yaml)
source VENV/uav/bin/activate
python src/ASPS_LIDAR/ASPS_LIDAR_TR.py --config configs/local/hardware.yaml
```

- Every flight writes a timestamped CSV to `flights/` (see `configs/logging.yaml`
  for the schema) plus video frames in `VIDEO_FOLDER` (default `results/`).
- `models/best.engine` is preferred on Jetson (TensorRT) — build it with
  `python model/export_trt.py --model models/best.pt --half` and point
  `YOLO_MODEL_PATH` at it.

---

## 3. Analyze a flight log

```bash
source VENV/uav/bin/activate
make analyze F=flights/<your_log>.csv        # metrics + figures to results/
# python tools/analyze_flight.py flights/<your_log>.csv
```

Metrics (yaw/pitch RMSE, MAE, P95, switch count, lock retention, ...) are computed
over `OFFBOARD` rows only. Historical CSVs with the old header names are auto-detected.

---

## 4. Model work

```bash
make train    DATA=/path/to/dataset          # train a new YOLO
make eval     DATA=/path/to/dataset          # mAP etc. on val split
make predict                                  # inference on images/video/webcam
make trt                                     # export .pt -> TensorRT .engine
```

Dataset layout: see `model/data.yaml` (set `$EAGLE_DATASET` or pass `DATA=`).

---

## 5. Ablation experiments (for the paper)

```bash
make ablation                              # runs the 4-cell SITL matrix, auto-analyzes
```

Results land in `experiments/ablation/results/comparison_summary.csv` and figures in
`results/`. See `experiments/ablation/run_ablation.py` to edit the matrix.

---

## Troubleshooting

- `ModuleNotFoundError: mavsdk` → `source VENV/uav/bin/activate` first.
- `Model not found: models/best.pt` → `make models`, or ask a maintainer for the
  private weights.
- No OFFBOARD command being sent → vehicle not engaged; switch mode and re-check
  the connection string (`udp://127.0.0.1:14550` for SITL).
- Wrong camera/LiDAR → check `configs/local/hardware.yaml` (your override wins).
