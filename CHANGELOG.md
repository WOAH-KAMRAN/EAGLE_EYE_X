# Changelog — EAGLE_EYE_X

All notable commits to the Eagle Eye X UAV Target Prioritization & Tracking System.

---

## 2026-08-08 — Repository restructure (team reproducibility)

- **All legacy scripts archived (never deleted)**: every superseded tracking and
  experimental script moved verbatim into `archive/` grouped by original location
  (`archive/root/`, `archive/src/`, `archive/experimental/`). See `archive/README.md`
  for the full lineage map.
- **Depth experiments consolidated** under `experiments/depth/` (was `MDE/` + `mdecalib.py`).
- **Production mainline** is now clearly `src/ASPS_LIDAR/` (`ASPS_LIDAR_TR.py` +
  `prioritization.py`); `CALLING.MD` renamed to `AGENTS.md`.
- **Removed junk**: `=1.71.0` (pip-log redirect), PostScript files in `src/`,
  `eeprom.bin`/`mav.parm`/`mav.tlog*`, `terrain/`. Duplicate model weights deduped
  into `models/`.
- **Security**: `visdrone` (Ultralytics Hub helper) stripped of its API key and
  archived.
- **`.gitignore`/`.gitattributes`** rewritten for the new layout (model weights,
  flight data, experiment outputs, local configs all stay untracked; LF line endings).
- **Research tooling added**: `tools/analyze_flight.py`, `experiments/ablation/`,
  `model/` toolkit, `configs/`, `Makefile`, `requirements*.txt`, `docs/`.

---

## 2025-06-22

- **`5ca887f`** — `Initial commit`
  - Created repository with `README.md`

---

## 2025-08-16

- **`4e6bcf7`** — `com1,withoutlidaryaw`
  - Added `yaw2.py` — Yaw-only control loop testing (673 lines)

---

## 2025-08-17

- **`19c7de1`** — `final script without lidar`
  - Added `WITHOUTLIDAR.py` — Mock YOLO variant with simulated targets (713 lines)

---

## 2025-08-26

- **`1b3cf7c`** — `gstreamer`
  - Added `wl.py` — Without-LiDAR variant with GStreamer support (empty initial commit)
- **`ea0741f`** — `gstreamer`
  - Populated `wl.py` with full implementation (855 lines)

---

## 2025-09-03

- **`0e34aba`** — `tensort boosted`
  - Added `upg3.py` — TensorRT boosted tracking variant (818 lines)
- **`bf1e8d8`** — `c`
  - Updated `upg3.py` — Refinements (45 insertions, 70 deletions)

---

## 2025-10-16

- **`d00f922`** — `Light Tracker usingpt`
  - Added `lightscr.py` — Lightweight tracking script with per-frame Kalman dt (520 lines)
- **`764b535`** — `modelpt`
  - Added `best.pt` — Custom-trained YOLO weights (18 MB)

---

## 2025-10-22

### README Updates

- **`173cbf8`** — `Update README.md`
  - Expanded README with system overview, core tech, and maintainers
- **`5d3a5cb`** — `Update README.md`
  - Added research notice, license section, and component descriptions
- **`c34f1eb`** — `Update README.md`
  - Minor formatting fixes

### Major Restructuring

- **`78adc59`** — `structured`
  - Moved all production scripts into `src/` directory:
    - `src/lastscr.py` (701 lines) — Safety-hardened flight-ready tracking
    - `src/enablederror.py` (765 lines) — lastscr.py + Lua failsafe heartbeat
    - `src/finalscript.py` (753 lines) — Earlier flight-ready variant
    - `src/WITHOUTLIDAR.py` (713 lines) — Mock YOLO variant
    - `src/lightscr.py` (520 lines) — Lightweight tracking
    - `src/trt.py` (105 lines) — Standalone TensorRT demo
    - `src/upg5repl.py` (862 lines) — TensorRT engine class + 4-class priorities
    - `src/upg6.py` (1333 lines) — MiDaS depth + distance PID + Kalman
    - `src/upg7.py` (1591 lines) — ZoeDepth metric depth + TensorRT YOLO
    - `src/upg9.py` (1434 lines) — upg7 + separate display thread

### Monocular Depth Estimation

- **`21275dc`** — `monodepth`
  - Added `MDE/` directory with depth estimation experiments:
    - `MDE/mdetest.py` (68 lines) — MiDaS standalone test
    - `MDE/monocular.py` (1310 lines) — Full tracking loop with MiDaS depth
    - `MDE/zoe.py` (82 lines) — Standalone ZoeDepth metric depth demo

### Cleanup & Security

- **`78c2f3e`** — `Remove secret files from repo`
  - Removed `best.pt` from tracked files (18 MB model weights)
- **`f452479`** — `Removered`
  - Removed `yaw2.py` from root
- **`bbee804`** — `Removered`
  - Removed `upg3.py` from root
- **`1795cc9`** — `Remove red`
  - Removed `WITHOUTLIDAR.py` from root
- **`217dba0`** — `Removered`
  - Removed `lightscr.py` from root

### Failsafe

- **`dc3c5bb`** — `failsafeaddn`
  - Added `failsafe/failsafe.lua` (157 lines) — Lua failsafe script for ArduPilot flight controller

---

## [Unreleased]

- Updated `.gitignore` — Comprehensive exclusion of VENV, models, flight data, binary artifacts, stray files, PostScript junk, and superseded root scripts
- Added `CHANGELOG.md` — This file
- Added `handout.md` — Comprehensive codebase handout with directory layout, script inventory, architecture diagrams, and quick start guide
- Added `agentdev.md` — Full agent development log: system architecture, Kalman filter design, PID tuning, ASPS design, TF Mini LiDAR integration, two-stage control, CSV logging, safety systems, SITL testing, ablation studies
- Added `src/lastscr_trt.py` — Production v2: per-frame Kalman dt, display thread, 4-class priorities, TensorRT .engine support
- Added `src/lidar_trt.py` — TF Mini LiDAR distance following with DistanceKalmanFilter, DistancePID, two-stage center-then-follow control
- Added `src/ASPS_LIDAR/ASPS_LIDAR_TR.py` — Adaptive Semantic Priority Selection + LiDAR integration (most advanced script, 949 lines)
- Added `src/ASPS_LIDAR/prioritization.py` — ASPS scoring engine: weighted mission hierarchy + confidence + distance + velocity with hysteresis
- Added `src/ASPS_LIDAR/README.md` — ASPS usage guide and configuration reference
- Added `src/ASPS_LIDAR/AGENTS.md` — AI agent instructions for the ASPS module
- Added `MDE/calib.py` — Focal length calibration tool (68 lines)
- Added `MDE/md2.py` through `MDE/md5.py` — MiDaS depth estimation experiments
- Added `MDE/midas_calibration.json` — MiDaS calibration parameters
- Added `mdecalib.py` — Interactive MiDaS depth calibration via Tkinter point-and-click interface
- Added `.vscode/settings.json` — VS Code Python interpreter configuration
- Modified `MDE/mdetest.py` — Minor updates
- Modified `src/WITHOUTLIDAR.py` — Minor updates
- Modified `src/lastscr.py` — Minor updates
