# Archive — Legacy Scripts (preserved, never deleted)

This directory preserves **every** superseded script from the Eagle Eye X development
lineage, verbatim. Files are grouped by their **original location** (`root/` = repo
root, `src/` = the old `src/` directory) so names stay byte-for-byte intact and the
research history is auditable.

> **Policy:** scripts in this folder are never deleted. They are kept for the
> research lineage (paper ablation context, reproducibility of earlier system
> versions) and for reference when porting features into the current mainline.

## Current mainline (the script to run)

`src/ASPS_LIDAR/ASPS_LIDAR_TR.py` — Adaptive Semantic Priority Selection (ASPS) +
TF Mini LiDAR distance following. Most advanced iteration; inherits per-frame Kalman
dt, display thread, 4-class priorities, two-stage center-then-follow control.

## Lineage map

### `root/` — original repo-root experiments

| File | What it was | Replaced by |
|---|---|---|
| `yaw2.py` | Yaw-only control loop test | upg1+ |
| `upg1.py` | Yaw/pitch alignment, PC webcam, no depth | upg2 |
| `upg2.py` | Jetson/PC variant, RPi Cam v2, YOLO ONNX placeholder | upg3 |
| `upg3.py` | TensorRT-boosted tracking | upg4 |
| `upg4.py` | Later iteration of TensorRT tracking | src/upg5repl |
| `idkupg.py` | TensorRT `.engine` tracking variant | src/upg5repl |
| `wl.py` | LiDAR distance following (GStreamer) | lidar_trt / ASPS_LIDAR |
| `WITHOUTLIDAR.py` | Mock-YOLO webcam variant | src/WITHOUTLIDAR |
| `mod1.py` | TensorRT + forward/backward following | src/upg5repl |
| `modelrun.py` | Minimal webcam YOLO demo | model/predict.py |
| `modelimg.py` | Batch inference `images/` → `results/` | model/predict.py |
| `mobilenet.py` | MobileNet-based follow (sample codebase) | — |

### `src/` — production-iteration lineage (as they evolved)

| File | What it was | Replaced by |
|---|---|---|
| `WITHOUTLIDAR.py` | Earlier no-depth flight variant | finalscript |
| `finalscript.py` | Flight-ready v1 | lastscr |
| `lastscr.py` | Safety-hardened flight-ready (RC override, timeouts, pre-flight) | enablederror |
| `enablederror.py` | lastscr + Lua failsafe heartbeat | lastscr_trt |
| `lightscr.py` | Lightweight variant, per-frame Kalman dt | lastscr_trt |
| `trt.py` | Standalone TensorRT demo | upg5repl |
| `upg5repl.py` | TensorRT engine class + 4-class priorities | upg6 |
| `upg6.py` | MiDaS depth + distance PID + Kalman | upg7 |
| `upg7.py` | ZoeDepth metric depth + TensorRT YOLO | upg9 |
| `upg9.py` | upg7 + separate display thread | lastscr_trt |
| `lastscr_trt.py` | Production v2 (KF dt, display thread, 4-class, TensorRT) | lidar_trt |
| `lidar_trt.py` | TF Mini LiDAR distance following (two-stage control) | ASPS_LIDAR_TR |

### `experimental/`

| File | What it was |
|---|---|
| `visdrone` | Ultralytics Hub model-training helper. **API key stripped** — rotate the original key before reuse. |

## Depth-estimation experiments

`../experiments/depth/` holds the monocular-depth lineage (`monocular.py`, `md2–md5`,
`mdetest.py`, `zoe.py`, `calib.py`, `mdecalib.py`, `midas_calibration.json`).
