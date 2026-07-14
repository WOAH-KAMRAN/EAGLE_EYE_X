# Eagle Eye X — Codebase Handout

**Project**: UAV Target Prioritization & Tracking System for NVIDIA Jetson Orin Nano  
**Repository**: `EAGLE_EYE_X/`

---

## Overview

Eagle Eye X is a physical AI system for multirotor UAVs that integrates object detection (YOLO), monocular depth estimation (MiDaS / ZoeDepth), and autonomous control (MAVSDK) into a single pipeline. It detects, prioritizes, and tracks high-value targets in real time using a single RGB camera.

**Core Tech Stack**: Python, PyTorch, Ultralytics YOLO, TensorRT, MiDaS / ZoeDepth, MAVSDK, OpenCV, NumPy

---

## Directory Layout

```
EAGLE_EYE_X/
├── src/                   # Main source scripts (production & iteration)
│   ├── lastscr.py         # Flight-ready tracking (safety-hardened)
│   ├── lightscr.py        # Lightweight version of tracking loop
│   ├── finalscript.py     # Earlier flight-ready variant
│   ├── enablederror.py    # lastscr.py + Lua failsafe heartbeat
│   ├── WITHOUTLIDAR.py    # Mock YOLO variant (no real model needed)
│   ├── trt.py             # Standalone TensorRT inference demo
│   ├── upg5repl.py        # Introduces TensorRT engine class
│   ├── upg6.py            # MiDaS depth + distance PID + Kalman
│   ├── upg7.py            # ZoeDepth metric depth + TensorRT YOLO
│   └── upg9.py            # upg7 + separate display thread
├── MDE/                   # Monocular Depth Estimation experiments
│   ├── zoe.py             # Standalone ZoeDepth metric depth demo
│   ├── monocular.py       # Full tracking loop (MiDaS depth)
│   ├── calib.py           # Focal length calibration tool
│   └── mdetest.py, md2-5.py  # MiDaS experiments
├── mdecalib.py            # MiDaS depth calibration tool (root)
├── modelrun.py            # Model runner
├── modelimg.py            # Model image inference
├── upd1.py - upd4.py      # Early experimental iterations
├── yaw2.py                # Yaw-only control test
├── wl.py                  # Early without-LiDAR variant
├── mobilenet.py           # Alternate classifier experiment
├── mod1.py                # Modular component test
├── idkupg.py              # Unknown upgrade experiment
├── best.pt                # Trained YOLO weights
├── flights/               # Flight data output (videos, logs)
├── flights1/              # Additional flight data
├── failsafe/              # Lua failsafe scripts (FC-side)
├── images/                # Test / calibration images
├── models/                # Model storage
├── results/               # Experiment results
└── VENV/                  # Python virtual environment
```

---

## Tier 1 — Core Production Scripts (Most Valuable)

### 1. `src/lastscr.py` (701 lines)
**The most flight-ready, safety-hardened script.** Integrates everything needed for autonomous UAV target tracking in production.

**Key features:**
- YOLOv8 object detection with configurable confidence/IOU thresholds
- Class-based target prioritization (custom `CLASS_PRIORITIES`)
- 1D Kalman Filters for yaw & pitch tracking (position + velocity state)
- Twin PID controllers for yaw and pitch alignment
- Altitude-hold PID (maintains `TARGET_ALTITUDE`)
- Pre-flight checks: connection, RC status, flight mode verification
- Safety: emergency stop (Ctrl+Q/Ctrl+C), RC override detection, altitude limits, telemetry timeouts, detection-loss hover
- CSV flight logging + MP4 video recording
- MAVSDK offboard control via `AttitudeRate` commands

### 2. `src/enablederror.py` (765 lines)
**Same as `lastscr.py` + Lua failsafe heartbeat.** The production choice when using companion-computer failsafe on the flight controller.

**Addition:**
- Background `heartbeat_task()` sends periodic timestamp updates to `SCR_USER1` MAVLink parameter
- Lua script on the FC monitors this parameter — if heartbeat stops, FC triggers failsafe (RTL/Land)
- Graceful heartbeat cancellation during emergency landing sequence

### 3. `src/lightscr.py` (520 lines)
**Clean, lightweight version** — good educational starting point for understanding the pipeline.

**Differences from lastscr.py:**
- Simpler PID (no output limits, no auto-dt)
- No altitude-hold PID
- No RC safety checks or preflight system
- Kalman filter `dt` updated per-frame instead of fixed
- GStreamer pipeline support for Jetson CSI cameras
- Configured for `best.pt` (custom trained model)

### 4. `src/upg7.py` (~1300+ lines)
**Latest feature iteration** — integrates ZoeDepth metric depth estimation + TensorRT YOLO + 3D forward velocity control.

**Key additions over Tier 1:**
- `ZoeDepthEstimator` class — metric depth in meters via `transformers` (HuggingFace)
- `MiDaSDepthEstimator` — fallback relative-to-metric depth conversion
- `MonocularDepthEstimator` — unified interface, auto-fallback ZoeDepth → MiDaS
- `DistanceKalmanFilter` — specialized Kalman for depth with noise handling
- Distance PID controller for forward/backward velocity (`VelocityNedYaw`)
- `YAW_CENTERED_THRESHOLD` — only moves forward when target is centered
- Configurable depth tuning: ROI scale, percentile, smoothing, median filter, confidence threshold

### 5. `src/upg9.py` (~1200+ lines)
**Latest UX iteration** — same as `upg7.py` but with a **separate display thread**.

**Key addition:**
- `display_thread_func()` runs OpenCV `imshow`/`waitKey` in a daemon thread
- Frame data passed via `queue.Queue` (maxsize=2, non-blocking)
- Prevents `cv2.waitKey()` from blocking the async drone control loop
- Thread-safe shutdown via `display_should_stop` event + `None` sentinel

---

## Tier 2 — Specialized / Component Scripts

### 6. `src/upg6.py` (~1300+ lines)
**Introduces monocular depth estimation** (MiDaS) for 2D-to-3D target following. Precursor to `upg7.py`.

- `MonocularDepthEstimator` with MiDaS_small model
- ROI-based depth sampling around target bounding box
- Depth map temporal smoothing
- Distance Kalman Filter + Distance PID
- `VelocityNedYaw` for forward velocity control

### 7. `src/upg5repl.py` (862 lines)
**Introduces TensorRT inference** for accelerated YOLO on Jetson.

- `TensorRTInference` class — loads `.engine` files, manages CUDA buffers
- `post_process_yolo_output()` — parses raw TensorRT output (center-format → corner-format, NMS)
- GStreamer CSI camera pipeline support
- Mock YOLO fallback

### 8. `src/trt.py` (105 lines)
**Minimal standalone TensorRT demo** — useful for validating TensorRT + PyCUDA setup in isolation.

- Loads `.engine` file, allocates buffers
- Continuous inference loop with camera display
- Press `q` to quit

### 9. `src/WITHOUTLIDAR.py` (713 lines)
**Earlier variant with mock YOLO** — no real model required. Includes sophisticated mock target simulation for testing prioritization logic.

- `mock_yolo_inference()` simulates 4 target classes with random movement/jitter
- Configurable detection probabilities per class
- Detailed CSV logging (I-term, KF estimates, raw errors)

---

## Tier 3 — Depth Estimation / Calibration (MDE/)

### 10. `MDE/zoe.py` (82 lines)
**Standalone ZoeDepth metric depth demo.** Minimal — just camera + ZoeDepth inference + side-by-side display.

### 11. `MDE/calib.py` (68 lines)
**Focal length calibration tool.** Given a known object size at known distance, computes `F_PIXEL` for converting pixel dimensions to real-world distances.

### 12. `mdecalib.py` (root, 167 lines)
**Interactive MiDaS depth calibration.** Click points in the depth map, enter real distances via Tkinter dialog, fits a linear model to calibrate MiDaS inverse-depth output.

### 13. `MDE/mdetest.py`, `md2.py`–`md5.py`
**MiDaS experiments** — rapid prototyping of different depth estimation approaches.

---

## Tier 4 — Iteration / Experiment Scripts (Root)

| Script | Purpose |
|---|---|
| `upg1.py`–`upg4.py` | Early experimental iterations: basic PID, mock YOLO, initial MAVSDK setup |
| `yaw2.py` | Yaw-only control loop testing |
| `wl.py` | Without-LiDAR variant |
| `modelrun.py` | Run a trained model (inference) |
| `modelimg.py` | Model inference on single images |
| `mobilenet.py` | MobileNet classifier experiment |
| `mod1.py` | Modular component test |
| `idkupg.py` | Unknown upgrade experiment |

---

## Files of Interest

| File | Description |
|---|---|
| `best.pt` | Custom-trained YOLO weights |
| `yolov8n.pt` | Base YOLOv8 nano model |
| `failsafe/` | Lua scripts for FC-side failsafe |
| `flights/`, `flights1/` | Flight data output directories |
| `MDE/midas_calibration.json` | MiDaS calibration parameters |
| `MDE/yolov8n.pt` | YOLO model for depth experiments |

---

## Architecture (Pipeline)

```
Camera Frame
    │
    ├──► YOLO Detection (PyTorch / TensorRT / Mock)
    │       └──► Class Prioritization → Best Target
    │
    ├──► Kalman Filter (Yaw + Pitch)
    │       └──► Smoothed target position
    │
    ├──► PID Controllers (Yaw + Pitch + Altitude / Distance)
    │       └──► AttitudeRate / VelocityNedYaw commands
    │
    ├──► MAVSDK → Drone (Offboard mode)
    │
    └──► CSV Logger + Video Writer + Display
```

**Optional depth pipeline (upg6/7/9):**
```
    └──► Monocular Depth (MiDaS / ZoeDepth)
            └──► ROI depth sampling → Distance estimate
            └──► Distance Kalman Filter
            └──► Distance PID → Forward velocity (VelocityNedYaw)
```

---

## Quick Start Recommendation

| Goal | Use |
|---|---|
| First-time understanding | `src/lightscr.py` (clean, simple) |
| Real flight deployment | `src/lastscr.py` or `src/enablederror.py` |
| Research / latest features | `src/upg7.py` or `src/upg9.py` |
| TensorRT setup test | `src/trt.py` |
| Depth calibration | `MDE/calib.py` or `mdecalib.py` |

---

## Maintainers

- Kamran Ahmad [@WOAH-KAMRAN]
- Melvin Joseph [@AVATAR3905]
- Ayushya Ranjan [@Dagger7164]
- Mentor: Dr. Sachin Gupta [@Sachinmait]
