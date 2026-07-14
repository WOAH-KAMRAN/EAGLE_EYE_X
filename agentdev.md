# Agent Development Log — EAGLE_EYE_X

> Comprehensive record of all changes, architecture decisions, system components, and operational knowledge for the UAV target tracking system.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Script Inventory](#3-script-inventory)
4. [lastscr_trt.py — Foundation Script](#4-lastscr_trtpy--foundation-script)
5. [lidar_trt.py — LiDAR Distance Following](#5-lidar_trtpy--lidar-distance-following)
6. [ASPS — Adaptive Semantic Priority Selection](#6-asps--adaptive-semantic-priority-selection)
7. [ASPS_LIDAR_TR.py — ASPS + LiDAR Integration](#7-asps_lidar_trpy--asps--lidar-integration)
8. [Kalman Filter Design](#8-kalman-filter-design)
9. [PID Controller Design](#9-pid-controller-design)
10. [Display Thread Architecture](#10-display-thread-architecture)
11. [TF Mini LiDAR Integration](#11-tf-mini-lidar-integration)
12. [Two-Stage Control Logic](#12-two-stage-control-logic)
13. [Distance Following (LiDAR/MDE)](#13-distance-following-lidarmde)
14. [CSV Logging Schema](#14-csv-logging-schema)
15. [Safety Systems](#15-safety-systems)
16. [SITL Testing Guide](#16-sitl-testing-guide)
17. [Ablation & Metrics](#17-ablation--metrics)
18. [Known Issues & Pitfalls](#18-known-issues--pitfalls)
19. [File Reference](#19-file-reference)

---

## 1. Project Overview

**EAGLE_EYE_X** is a real-time UAV target tracking system for the Jetson Orin Nano. It uses:

- **YOLO** (Ultralytics) for object detection
- **Kalman filters** for yaw/pitch/distance smoothing
- **PID controllers** for drone attitude and velocity control
- **MAVSDK** for drone communication (MAVLink)
- **OpenCV** for camera capture and display
- **TF Mini LiDAR** (optional) for distance measurement
- **ASPS** for adaptive target scoring (mission hierarchy + runtime refinement)

All scripts connect to a drone via `udpin://127.0.0.1:14550` (SITL) or `serial:///dev/ttyTHS1:921600` (real hardware).

### Key Design Principles

- **Safety first**: RC override detection, altitude limits, telemetry timeouts, emergency stop, landing sequence in `finally` block
- **Non-blocking display**: `cv2.imshow`/`cv2.waitKey` runs in a dedicated thread so the async control loop is never stalled
- **Per-frame Kalman dt**: The Kalman filter's state transition matrix is updated every frame with the real frame-to-frame interval, not a hardcoded 30 FPS assumption
- **Mission hierarchy preserved**: CLASS_PRIORITIES remain the dominant factor in target selection; runtime sensors only refine, not override
- **Ablation-ready**: Every new feature has a toggle (`USE_ASPS`, `USE_REAL_LIDAR`) so baselines can be compared

---

## 2. System Architecture

### 2.1 High-Level Data Flow

```
Camera ──→ YOLO ──→ detections ──→ ASPS ──→ best target
                  (class, conf,     │
                   bbox xyxy)       │
                                    ▼
                           Kalman Filter (yaw/pitch)
                           │         │
                           ▼         ▼
                     Yaw PID    Pitch PID
                           │         │
                           ▼         ▼
                     AttitudeRate / VelocityNedYaw
                           │
                           ▼
                        MAVSDK ──→ Drone
                           │
                           ▼
                     Telemetry (alt, mode, RC)
```

### 2.2 Module Responsibilities

| Module | Responsibility |
|---|---|
| **YOLO** | Object detection: class ID, confidence, bounding box |
| **ASPS** | Target scoring: mission hierarchy + confidence + distance + velocity |
| **Kalman Filter** | Smooth yaw/pixel position; remove bounding-box jitter |
| **Distance Kalman Filter** | Smooth LiDAR/depth distance reading over time |
| **PID (yaw)** | Yaw rate command to center target horizontally |
| **PID (pitch)** | Pitch rate command to center target vertically |
| **PID (altitude)** | Thrust adjustment to maintain target altitude |
| **PID (distance)** | Forward velocity command to maintain follow distance |
| **Display Thread** | Separate thread running `cv2.imshow` + `cv2.waitKey` |
| **TF Mini LiDAR** | Serial interface reading distance from laser rangefinder |
| **MAVSDK** | Drone connection, telemetry, offboard commands |

### 2.3 Threading Model

```
┌─────────────────────────────────────────────┐
│  Main Thread (async)                        │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐ │
│  │ camera    │→ │ YOLO     │→ │ control  │ │
│  │ read()    │  │ detect() │  │ loop     │ │
│  └───────────┘  └──────────┘  └────┬─────┘ │
│                                     │       │
│              frame.copy() ──────────┼────── │
│                                     │       │
│  ┌──────────────────────────────────▼────┐  │
│  │           display_queue.put()        │  │
│  └───────────────────────────────────────┘  │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│  Display Thread (daemon)                    │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐ │
│  │ queue.get │→ │ imshow() │→ │ waitKey()│ │
│  └───────────┘  └──────────┘  └──────────┘ │
└─────────────────────────────────────────────┘
```

The display thread communicates with the main thread via:
- `display_queue` (`queue.Queue(maxsize=2)`) — frame transfer (non-blocking put, blocking get with 0.1s timeout)
- `display_should_stop` (`threading.Event`) — shutdown signal

---

## 3. Script Inventory

### 3.1 Source Scripts (`src/`)

| Script | Lines | Purpose | Status |
|---|---|---|---|
| `lastscr.py` | 701 | Original base script (untouched reference) | Stable |
| `lastscr_trt.py` | 819 | Per-frame KF dt, display thread, 4-class priorities, TRT-ready | Active |
| `lidar_trt.py` | 931 | LiDAR distance following (TF Mini) | Active |
| `ASPS_LIDAR/ASPS_LIDAR_TR.py` | 949 | ASPS + LiDAR following (with ablation toggle) | Active |
| `ASPS_LIDAR/prioritization.py` | 160 | ASPS engine module | Active |
| `upg7.py` | 1591 | ZoeDepth + YOLO depth following (reference) | Reference |
| `upg9.py` | 1434 | Display thread pattern source (reference) | Reference |
| `lightscr.py` | 520 | Per-frame dt Kalman filter pattern (reference) | Reference |
| `upg5repl.py` | 862 | 4-class priorities + raw TensorRT class (reference) | Reference |

### 3.2 Key Differences Between Scripts

| Feature | lastscr.py | lastscr_trt.py | lidar_trt.py | ASPS_LIDAR_TR.py |
|---|---|---|---|---|
| Kalman dt | Hardcoded 0.033s | Per-frame `update_dt()` | Per-frame `update_dt()` | Per-frame `update_dt()` |
| Display | Main thread `waitKey` | Display thread | Display thread | Display thread |
| Class priorities | `{0: 1}` (placeholder) | `{2:4, 1:3, 3:2, 0:1}` | Same | Same + ASPS scoring |
| Distance source | None | None | TF Mini LiDAR | TF Mini LiDAR |
| Distance KF | None | None | `DistanceKalmanFilter` | `DistanceKalmanFilter` |
| Distance PID | None | None | `dist_pid` | `dist_pid` |
| Forward control | None | None | `VelocityNedYaw` | `VelocityNedYaw` |
| Target selection | `max(priority)` | `max(priority)` | `max(priority)` | ASPS or legacy |
| CSV columns | 9 | 10 | 15 | 20 |
| Signal handler | Custom (broken) | None (default) | None (default) | None (default) |

---

## 4. lastscr_trt.py — Foundation Script

### 4.1 What It Is

A production-ready rewrite of `lastscr.py` that fixes three concrete issues:

1. **Hardcoded Kalman dt** → per-frame `update_dt()` method
2. **Blocking display** → separate display thread with `queue.Queue`
3. **Single class priority** → 4-class scheme from `upg5repl.py`

### 4.2 What Changed from lastscr.py

**KalmanFilter class (lines 160-204)**:
- Constructor now takes `dt` as first parameter: `KalmanFilter(dt=0.033, Q=0.1, R=10, initial_pos=0.0)`
- `self.A` is computed from `dt`: `np.array([[1, dt], [0, 1]])`
- `self.Q_matrix` is computed from `dt`: `Q * [[0.25*dt**4, 0.5*dt**3], [0.5*dt**3, dt**2]]`
- New method `update_dt(dt)` recomputes `A` and `Q_matrix` when `dt` changes
- Old `self.F` matrix (hardcoded `[[1, 0.033], [0, 1]]`) is gone

**Display thread (lines 258-298)**:
- New globals: `display_queue = queue.Queue(maxsize=2)`, `display_should_stop = threading.Event()`
- New function `display_thread_func()` — runs `cv2.imshow`/`cv2.waitKey` in a daemon thread
- Main loop: `display_queue.put_nowait(frame.copy())` instead of `cv2.imshow(...)`
- Key check: `if display_should_stop.is_set(): break` — triggered by 'q' key in display thread
- Shutdown in `finally`: `display_should_stop.set()`, `display_queue.put(None)`, `display_thread.join(timeout=2.0)`

**Class priorities (line 53-58)**:
```python
CLASS_PRIORITIES = {2: 4, 1: 3, 3: 2, 0: 1}
```

**Ctrl+C handling (lines 812-819)**:
- Removed custom `signal_handler()` function that blocked `KeyboardInterrupt`
- Removed `signal.signal(signal.SIGINT, signal_handler)`
- `except KeyboardInterrupt` now sets `EMERGENCY_STOP = True` before printing
- Python's default SIGINT → `KeyboardInterrupt` → `asyncio.run()` cancels pending coroutines → `finally` runs landing sequence

**Documentation added**:
- TensorRT Option A (Ultralytics `.engine` path) documented as a comment
- Per-frame dt logged to CSV and displayed on screen: `f"dt: {dt*1000:.0f}ms"`

### 4.3 Key Bug Fix: Ctrl+C

The original `signal_handler()` in `lastscr.py` set `EMERGENCY_STOP = True` but never raised `KeyboardInterrupt`. Since `asyncio.run()` catches `KeyboardInterrupt`, the custom handler prevented it from being raised. The fix was to remove the handler entirely and let Python's default behavior raise `KeyboardInterrupt`, which `asyncio.run()` handles cleanly by cancelling all pending tasks.

---

## 5. lidar_trt.py — LiDAR Distance Following

### 5.1 What It Is

A version of `lastscr_trt.py` that adds TF Mini LiDAR distance measurement and forward velocity control, creating a complete "center → measure → follow" pipeline.

### 5.2 New Components

**import serial (lines 39-44)**: Optional import; mock mode works without pyserial.

**LiDAR config (lines 90-95)**:
```python
USE_REAL_LIDAR = False      # False = mock mode
LIDAR_PORT = '/dev/ttyUSB0'
LIDAR_BAUDRATE = 115200
LIDAR_MIN_RANGE_M = 0.1
LIDAR_MAX_RANGE_M = 12.0
```

**Distance following config (lines 97-110)**:
```python
TARGET_DISTANCE_M = 2.0
DISTANCE_DEAD_ZONE_M = 0.15
YAW_CENTERED_THRESHOLD = 30
MAX_FORWARD_SPEED_MPS = 2.0
KP_DIST = 1.0
KI_DIST = 0.05
KD_DIST = 0.3
Q_DIST_KF = 0.1
R_DIST_KF = 3.0  # LiDAR is accurate — low noise
```

**class DistanceKalmanFilter (lines 207-246)**: Identical structure to yaw/pitch KF but tuned for 1D distance. Measurement noise `R=3.0` (compared to `R=10` for yaw/pitch) because LiDAR is more accurate.

**class TFMiniLidar (lines 282-361)**:
- `__init__`: configures real vs mock mode
- `initialize()`: opens serial port or prints mock message
- `read_distance()`: returns meters or None
- `_read_real()`: parses 9-byte TF Mini serial frames (header `YY`, little-endian cm distance, strength validation)
- `_read_mock()`: simulates distance near `TARGET_DISTANCE_M` with Gaussian noise + drift
- `cleanup()`: closes serial port

### 5.3 Two-Stage Control Logic (lines 709-745 in main loop)

```
if best_det is not None:
    ┌─────────────────────────────────────┐
    │ Step 1: Center target               │
    │ Yaw PID:  yaw_err → yaw_cmd        │
    │ Pitch PID: pitch_err → pitch_cmd    │
    └─────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │ Step 2: Check centered?             │
    │ yaw_err <= YAW_CENTERED_THRESHOLD   │
    └─────────────────────────────────────┘
           │                  │
         YES                  NO
           │                  │
           ▼                  ▼
    ┌──────────────┐   ┌──────────────┐
    │ Read LiDAR   │   │ AttitudeRate │
    │ DistKF update│   │ (keep        │
    │ DistancePID  │   │  centering)  │
    │ VelocityNed  │   └──────────────┘
    │ Yaw(fwd,...) │
    └──────────────┘
```

When the target is centered (yaw error ≤ 30 pixels), the script switches from `AttitudeRate` control (yaw/pitch rates + thrust) to `VelocityNedYaw` (forward velocity + yaw rate). This two-mode approach is critical because:

- **AttitudeRate** works well for fine centering adjustments
- **VelocityNedYaw** is needed for forward/backward movement without fighting the altitude controller
- The FC maintains altitude automatically in velocity mode (down velocity = 0)

### 5.4 CSV Logging (lines 579-586)

Extended from 10 to 15 columns:
```
Time, Mode, Alt, Alt_Err,
Yaw_Cmd, Pitch_Cmd, Raw_LiDAR_m, KF_Dist_m,
Dist_Err_m, Fwd_Vel_mps, Target_Centered,
Class, Conf, RC_OK, KF_dt
```

- `Raw_LiDAR_m`: most recent LiDAR reading in meters
- `KF_Dist_m`: Kalman-filtered distance estimate
- `Dist_Err_m`: `KF_Dist - TARGET_DISTANCE_M`
- `Fwd_Vel_mps`: forward velocity command in m/s
- `Target_Centered`: boolean (0/1) indicating centered state

### 5.5 UI Overlay (lines 816-840)

```
TRACKING: 2              ← class ID
Alt: 2.1m (T:2.0m)       ← altitude info
FOLLOW | LiDAR: 2.15m    ← mode + raw LiDAR
DistKF: 2.12m | Fwd: 0.32m/s  ← filtered distance + velocity
Yaw: 12.3 Pitch: 5.1     ← control commands
dt: 998ms                ← per-frame interval
```

When centered: a green circle of radius `YAW_CENTERED_THRESHOLD` is drawn around center.

---

## 6. ASPS — Adaptive Semantic Priority Selection

### 6.1 Concept

ASPS upgrades the fixed `CLASS_PRIORITIES` dictionary into a weighted scoring system that considers runtime information while keeping the mission hierarchy dominant.

**Formula**:
```
P_final = ALPHA * P_base + BETA * Confidence + GAMMA * Distance + DELTA * Velocity
```

**Default weights** (sum = 1.0):
```
ALPHA = 0.60   ← mission hierarchy (dominant)
BETA  = 0.20   ← detector confidence
GAMMA = 0.15   ← target proximity (from LiDAR)
DELTA = 0.05   ← target motion (minor)
```

### 6.2 Component Definitions

**Base priority (P_base)**:
```python
normalized = CLASS_PRIORITIES[class_id] / max(CLASS_PRIORITIES.values())
# Class 2 → 1.00  (4/4)
# Class 1 → 0.75  (3/4)
# Class 3 → 0.50  (2/4)
# Class 0 → 0.25  (1/4)
# Weighted contribution: ALPHA * normalized = 0.60 * normalized
```

**Confidence**: Direct detector confidence in [0, 1]. Weighted contribution: `BETA * conf`.

**Distance score**:
```python
min(1.0, 1.0 / max(distance_m, 1.0))
# 1m  → 1.00
# 2m  → 0.50
# 5m  → 0.20
# 10m → 0.10
# None → 0.00  (distance not available)
```
Weighted contribution: `GAMMA * distance_score`.

**Velocity score**: Computed from bounding-box center displacement per frame, exponentially smoothed, normalized to [0, 1]. Weighted contribution: `DELTA * velocity_score`.

### 6.3 Velocity Tracking

The `VelocityTracker` class stores per-class_id `(cx, cy, timestamp, filtered_velocity)`.

```python
# Each frame:
raw_velocity = sqrt(dx*dx + dy*dy) / dt
filtered_velocity = 0.8 * previous_velocity + 0.2 * raw_velocity

# Normalized score:
score = min(filtered_velocity / 500.0, 1.0)
# 500 px/s ≈ full motion (clamped)
```

Keyed by `class_id`, not by persistent object ID. Two objects of the same class share velocity state. This is a simplification — fine for single-target tracking but means ASPS cannot distinguish between two moving targets of the same class.

### 6.4 Hysteresis

Prevents rapid target switching between similarly-scored detections:

```python
SWITCH_THRESHOLD = 0.15
```

Logic: only switch to a different-class target when `new_score > current_score + 0.15`. When the current target's class is no longer detected, the switch is allowed unconditionally.

### 6.5 Ablation Toggle

```python
USE_ASPS = True   # in prioritization.py
```

When `False`, `ASPSPrioritySelector.select()` falls back to:
```python
best = max(detections, key=lambda d: CLASS_PRIORITIES.get(d['cls'], 0))
```
This is identical to the original `lidar_trt.py` selection logic, enabling clean comparison between fixed-priority and ASPS modes.

---

## 7. ASPS_LIDAR_TR.py — ASPS + LiDAR Integration

### 7.1 What It Is

A copy of `lidar_trt.py` with the target selection loop replaced by `ASPSPrioritySelector`. This is the most advanced script in the project — combines all features.

### 7.2 Changes from lidar_trt.py

**Import** (line 5 from bottom):
```python
from prioritization import ASPSPrioritySelector, USE_ASPS
```

**ASPS init** (after LiDAR init):
```python
selector = ASPSPrioritySelector()
```

**Target selection replacement** — this 8-line block in `lidar_trt.py`:
```python
best_det = None
best_priority = -1
for det in detections:
    priority = CLASS_PRIORITIES.get(det['cls'], 0)
    if priority > best_priority:
        best_priority = priority
        best_det = det
```
becomes this 4-line block:
```python
def distance_fn(det):
    return raw_lidar_m if target_centered else None
best_det, components = selector.select(
    detections, time.time(), CLASS_PRIORITIES, distance_fn
)
```

**Component breakdown extraction**:
```python
if components is not None:
    comp_base = components['base_priority']
    comp_conf = components['confidence']
    comp_dist = components['distance']
    comp_vel  = components['velocity']
    final_score = comp_base + comp_conf + comp_dist + comp_vel
else:
    comp_base = comp_conf = comp_dist = comp_vel = 0.0
    final_score = 0.0
```

**CSV columns** (extended from 15 to 20):
```
'Time','Mode','Alt','Alt_Err',
'Yaw_Cmd','Pitch_Cmd','Raw_LiDAR_m','KF_Dist_m',
'Dist_Err_m','Fwd_Vel_mps','Target_Centered',
'Class','Conf','RC_OK','KF_dt',
'BaseP','ConfP','DistP','VelP','FinalP'
```

**UI overlay** — shows ASPS component breakdown:
```
Score: 0.83  Base:0.60 Conf:0.19
Dist:0.08 Vel:0.04  Fwd:0.32m/s
```

**Bounding box color**: ASPS-selected target = cyan `(255, 255, 0)`, others = red `(100, 100, 255)`.

**ASPS indicator**: Shows "ASPS: ON" (green) or "ASPS: OFF" (gray) in top-right corner.

**Selector reset**: `selector.reset()` called in the STANDBY branch when not in OFFBOARD mode.

---

## 8. Kalman Filter Design

### 8.1 Yaw/Pitch Kalman Filter

**State**: `[position (pixels), velocity (pixels/sec)]`

**Per-frame dt update**:
```python
def update_dt(self, dt):
    self.dt = dt
    self.A = np.array([[1, dt], [0, 1]])
    self.Q_matrix = np.array([[0.25*dt**4, 0.5*dt**3],
                              [0.5*dt**3, dt**2]]) * self.Q
```

**Predict**:
```python
self.x = self.A @ self.x                 # x_k = A * x_{k-1}
self.P = self.A @ self.P @ self.A.T + self.Q_matrix  # P_k = A*P_{k-1}*A^T + Q
```

**Update**:
```python
y = measurement - self.H @ self.x         # innovation
S = self.H @ self.P @ self.H.T + self.R   # innovation covariance
K = self.P @ self.H.T @ inv(S)            # Kalman gain
self.x = self.x + K @ y                   # state update
self.P = (I - K @ self.H) @ self.P        # covariance update
```

### 8.2 Distance Kalman Filter

Identical structure but:
- State: `[distance (m), velocity (m/s)]`
- `R = 3.0` for LiDAR (accurate), `R = 10.0` for monocular depth (noisier)
- Initial `P = 100` instead of `1000` for yaw/pitch (less initial uncertainty)
- `update()` accepts `None` measurement (skips update if distance unavailable)

### 8.3 Why Per-frame dt Matters

The original `lastscr.py` hardcoded `dt = 0.033s` (30 FPS). When the actual loop runs at 1 FPS (as seen in flight logs), the filter predicts with the wrong model:

```
F matrix with dt=0.033:  [[1, 0.033], [0, 1]]
Actual dt=1.0s:          [[1, 1.0],   [0, 1]]
```

The velocity gain term is wrong by 30×, causing the position estimate to diverge during the predict step. The `update_dt(dt)` call before `predict()` fixes this.

---

## 9. PID Controller Design

### 9.1 Standard Form

All PIDs use the same class:
```python
PID(Kp, Ki, Kd, output_limit=None, integrator_limit=100)
```

**Update**:
```python
P = Kp * error
I += Ki * error * dt
D = Kd * (error - last_error) / dt
output = P + I + D
```

**Clamp**: output clamped to `[-output_limit, output_limit]`
**Reset**: zeroes `I_term`, `last_error`, `last_time`

### 9.2 PID Tuning (All Scripts)

| Controller | Kp | Ki | Kd | Output Limit | Notes |
|---|---|---|---|---|---|
| Yaw (centering) | 0.03 | 0.001 | 0.01 | ±30°/s | Conservative for tracking |
| Pitch (centering) | 0.02 | 0.001 | 0.01 | ±15°/s | Conservative |
| Altitude hold | 0.5 | 0.05 | 0.1 | ±0.3 thrust | Oscillates in logs — needs tuning |
| Distance (LiDAR) | 1.0 | 0.05 | 0.3 | ±2.0 m/s | Responsive for LiDAR |
| Distance (depth) | 0.8 | 0.03 | 0.2 | ±2.0 m/s | More conservative (noisier sensor) |

### 9.3 Ziegler-Nichols Starting Point

If tuning from scratch:
1. Set Ki = Kd = 0
2. Increase Kp until steady oscillation
3. Ku = that Kp, Tu = oscillation period
4. Kp = 0.6 * Ku, Ki = 2 * Kp / Tu, Kd = Kp * Tu / 8

---

## 10. Display Thread Architecture

### 10.1 Why Separate Thread

`cv2.waitKey(1)` in the main loop blocks Python's event loop because OpenCV needs to process window events. On some systems, `waitKey` can block for 10-50ms even with a 1ms argument. This stalls the `asyncio` event loop, delaying telemetry reads and drone commands.

The display thread isolates this blocking call:

```python
# Main thread
async def tracking_loop():
    ...
    try:
        display_queue.put_nowait(frame.copy())  # 0-blocking
    except queue.Full:
        pass
    ...

# Display thread (separate daemon thread)
def display_thread_func():
    ...
    while not display_should_stop.is_set():
        try:
            frame = display_queue.get(timeout=0.1)
            cv2.imshow("UAV Tracking", frame)
            cv2.waitKey(1)  # Blocking is OK here
        except queue.Empty:
            cv2.waitKey(1)
            continue
```

### 10.2 Thread Communication Protocol

```
Main → Display: frame.copy() via queue.put_nowait()
                 None via queue.put() (shutdown sentinel)
Display → Main: display_should_stop.set() when 'q' pressed
                 (checked via if display_should_stop.is_set(): break)
```

- Queue maxsize=2: if the display thread is slower than the main loop, old frames are dropped
- `.copy()` is critical: without it, the main loop continues to draw on the same frame object being displayed

### 10.3 Shutdown Sequence

```
1. Main loop catches KeyboardInterrupt / ends
2. finally block:
   a. Stop offboard, land, disarm
   b. display_should_stop.set()
   c. display_queue.put(None, timeout=1.0)
   d. display_thread.join(timeout=2.0)
   e. Check if alive → print warning
   f. cap.release(), video_writer.release(), log_file.close()
```

---

## 11. TF Mini LiDAR Integration

### 11.1 Hardware Protocol

The TF Mini LiDAR communicates over UART (USB serial) at 115200 baud. It sends continuous 9-byte frames:

```
Byte 0: 'Y' (0x59) — header byte 1
Byte 1: 'Y' (0x59) — header byte 2
Byte 2-3: Distance (little-endian uint16, in centimeters)
Byte 4-5: Strength (little-endian uint16)
Byte 6: Reserved
Byte 7-8: Checksum (not validated in current code)
```

### 11.2 Real Mode

```python
def _read_real(self):
    self.serial_port.flushInput()
    while self.serial_port.in_waiting >= 9:
        if self.serial_port.read(1) == b'Y':
            if self.serial_port.read(1) == b'Y':
                frame_data = self.serial_port.read(7)
                dist_cm = int.from_bytes(frame_data[0:2], 'little')
                strength = int.from_bytes(frame_data[2:4], 'little')
                dist_m = dist_cm / 100.0
                if 0.1 <= dist_m <= 12.0 and strength > 10:
                    self.last_valid_distance = dist_m
                    return dist_m
    return None  # No valid frame
```

### 11.3 Mock Mode

```python
def _read_mock(self):
    if self.last_valid_distance is None:
        base = TARGET_DISTANCE_M + random.uniform(-1.0, 1.0)
    else:
        base = self.last_valid_distance + random.uniform(-0.05, 0.05)
    noise = random.uniform(-0.02, 0.02)
    return clip(base + noise, 0.1, 12.0)
```

The mock mode simulates:
- Initial acquisition: random distance near target (2m ± 1m)
- Frame-to-frame drift: ±5cm
- Measurement noise: ±2cm
- Range limit: 0.1–12m

This allows full testing in SITL without any LiDAR hardware.

### 11.4 Alignment Requirement

The TF Mini beam is ~2cm wide at 12m. For accurate readings, the target must be centered in the camera frame. This is enforced by the `YAW_CENTERED_THRESHOLD` check — LiDAR is only read when `abs(yaw_err) <= 30px`.

The LiDAR is assumed to be mounted parallel to the camera (boresighted). Physically aligning the LiDAR to the camera's optical axis is a hardware prerequisite.

---

## 12. Two-Stage Control Logic

### 12.1 Overview

```
                   ┌─────────────┐
                   │ YOLO detect │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │ Any target? │──No──→ HOVER / SEARCH (AttitudeRate, zero rates)
                   └──────┬──────┘
                          │ Yes
                   ┌──────▼──────┐
                   │ Center      │
                   │ (yaw_pid +  │──Both PIDs active
                   │  pitch_pid) │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │ Centered?   │
                   │ (yaw_err    │
                   │  <= 30px)   │
                   └──────┬──────┘
                     NO       YES
                      │        │
                      ▼        ▼
               ┌────────┐ ┌─────────────────┐
               │ Keep   │ │ Read LiDAR       │
               │ center │ │ DistKF.update() │
               │ (Atti- │ │ DistPID.update()│
               │ tude   │ │ VelNedYaw(fwd) │
               │ Rate)  │ └─────────────────┘
               └────────┘
```

### 12.2 AttitudeRate Mode

Used when the target is not centered:
```python
await drone.offboard.set_attitude_rate(AttitudeRate(
    0.0,                           # roll (0)
    np.deg2rad(-pitch_cmd),        # pitch rate (rad/s)
    np.deg2rad(yaw_cmd),            # yaw rate (rad/s)
    thrust                          # throttle [0, 1]
))
```

- Thrust is computed from altitude PID: `thrust = clip(HOVER_THRUST + alt_pid.update(alt_err), 0.3, 0.8)`
- This mode allows fine centering but cannot move forward/backward efficiently

### 12.3 VelocityNedYaw Mode

Used when the target is centered:
```python
await drone.offboard.set_velocity_ned(VelocityNedYaw(
    forward_velocity,    # North (+ = forward)
    0.0,                 # East (0 = no lateral)
    0.0,                 # Down (0 = maintain altitude)
    np.deg2rad(yaw_cmd)  # Yaw rate (rad/s)
))
```

- Forward velocity is the output of the distance PID: `dist_pid.update(dist_err)`
- Altitude is maintained by the FC (down velocity = 0)
- The FC's internal altitude controller keeps the drone level

### 12.4 Why Two Modes?

SITL/ArduPilot handles `set_velocity_ned` differently from `set_attitude_rate`:

| Mode | Altitude handling | Lateral movement | Best for |
|---|---|---|---|
| AttitudeRate | Manual (thrust) | Yes | Fine centering, hovering |
| VelocityNedYaw | Automatic (FC) | Yes | Forward movement, distance following |

Switching to VelocityNedYaw for forward movement avoids fighting the altitude controller — the FC maintains altitude internally while we control forward velocity and yaw rate.

---

## 13. Distance Following (LiDAR/MDE)

### 13.1 Pipeline

```
LiDAR reading / depth sample → DistanceKalmanFilter.update() → DistancePID.update() → forward_velocity → VelocityNedYaw
```

### 13.2 Distance Kalman Filter

```
State: [distance (m), velocity (m/s)]
Measurement: LiDAR distance (meters)
R: 3.0  (LiDAR is accurate)
  10.0 (MDE is noisier)
```

The KF smooths the noisy LiDAR/depth readings. When the target disappears briefly, the KF's predict step continues to provide a distance estimate (with increasing uncertainty).

### 13.3 Distance PID

```
Input: dist_err = KF_distance - TARGET_DISTANCE_M
Output: forward_velocity (m/s)
Output limit: ±MAX_FORWARD_SPEED_MPS (±2.0 m/s)
Dead zone: ±0.15m
```

When `abs(dist_err) <= DISTANCE_DEAD_ZONE_M`, the PID is reset and velocity = 0 — the drone hovers at the desired follow distance.

### 13.4 LiDAR vs MDE Comparison

| Aspect | LiDAR (TF Mini) | MDE (ZoeDepth/MiDaS) |
|---|---|---|
| Hardware | $20-30 USB sensor | None (GPU compute) |
| Accuracy | ±1-3cm | ±20-30% of range |
| Latency | ~1ms | ~50-200ms |
| GPU memory | 0 | +2-3GB |
| Field of view | Single point | Dense 640×480 map |
| Night ops | Yes | No (needs RGB) |
| DistKF R | 3.0 | 10.0 |
| PID Kp_dist | 1.0 | 0.8 |

---

## 14. CSV Logging Schema

### 14.1 Column Evolution

| Script | Columns | Purpose |
|---|---|---|
| lastscr_trt.py | 10 | Time, Mode, Alt, Alt_Err, Yaw_Cmd, Pitch_Cmd, Class, Conf, RC_OK, KF_dt |
| lidar_trt.py | 15 | + Raw_LiDAR_m, KF_Dist_m, Dist_Err_m, Fwd_Vel_mps, Target_Centered |
| ASPS_LIDAR_TR.py | 20 | + BaseP, ConfP, DistP, VelP, FinalP |

### 14.2 Full Schema (ASPS_LIDAR_TR.py)

```
Time             float  → time.time() at log write
Mode             str    → flight mode name
Alt              float  → relative altitude (m)
Alt_Err          float  → target - actual altitude (m)
Yaw_Cmd          float  → yaw rate command (deg/s)
Pitch_Cmd        float  → pitch rate command (deg/s)
Raw_LiDAR_m      float  → raw LiDAR distance (m)
KF_Dist_m        float  → Kalman-filtered distance (m)
Dist_Err_m       float  → KF_dist - target_distance (m)
Fwd_Vel_mps      float  → forward velocity command (m/s)
Target_Centered  int    → 1 if yaw_err <= threshold, else 0
Class            int    → tracked class ID (-1 if none)
Conf             float  → detector confidence
RC_OK            bool   → RC connection status
KF_dt            float  → frame-to-frame interval (s)
BaseP            float  → ASPS base priority component
ConfP            float  → ASPS confidence component
DistP            float  → ASPS distance component
VelP             float  → ASPS velocity component
FinalP           float  → ASPS total score (sum of components)
```

### 14.3 Log Analysis Recipe

```python
import pandas as pd
df = pd.read_csv('log_20260703_011455.csv')

# Tracking duration
print(f"Duration: {df['Time'].iloc[-1] - df['Time'].iloc[0]:.1f}s")

# Target switch frequency
switches = (df['Class'].diff() != 0).sum()
print(f"Target switches: {switches}")

# Mean distance error
print(f"Mean dist error: {df['Dist_Err_m'].mean():.3f}m")

# Time spent in follow mode vs centering mode
follow_ratio = df['Target_Centered'].mean()
print(f"Follow ratio: {follow_ratio:.1%}")
```

---

## 15. Safety Systems

### 15.1 Emergency Stop

- `EMERGENCY_STOP = True` triggers break from main loop → landing sequence
- Set by: `KeyboardInterrupt` (Ctrl+C), 'q' key in display thread
- Checked: at start of each frame, during takeoff loop

### 15.2 RC Override

- Telemetry includes RC status check
- If RC lost (`rc_ok == False`): immediately breaks → lands
- This allows the pilot to take control at any time by moving sticks

### 15.3 Altitude Limits

```
MAX_ALTITUDE = 10.0m   → hard ceiling
MIN_ALTITUDE = 0.5m    → don't fly into ground
TARGET_ALTITUDE = 2.0m → default follow height
```

Break → land if exceeded.

### 15.4 Telemetry Timeout

```python
TELEMETRY_TIMEOUT = 2.0  # seconds
```

Each `await` for telemetry has a `wait_for` timeout. If telemetry is lost for 2s, the script lands. This prevents fly-aways if the MAVLink link drops.

### 15.5 Detection Timeout

```python
DETECTION_TIMEOUT = 5.0  # max time without detection before hover
```

If no target detected for 5s: reset PIDs, zero commands → drone hovers in place. Prevents chasing ghosts when YOLO misses frames.

### 15.6 Takeoff Timeout

```python
TAKEOFF_TIMEOUT = 20.0  # max time to reach altitude
```

If takeoff takes >20s, abort and land.

### 15.7 Landing Sequence (finally block)

Always executes, regardless of how the loop exits:
1. Stop offboard mode
2. `action.land()` — ArduPilot handles descent
3. Wait 8 seconds for landing
4. `action.disarm()` — disarm motors
5. Signal display thread to stop
6. Release camera, video writer, log file

---

## 16. SITL Testing Guide

### 16.1 Starting SITL

```bash
# Terminal 1: Start ArduPilot SITL
cd ~/ardupilot
sim_vehicle.py -v ArduCopter --map --console
```

This starts:
- ArduCopter SITL binary (software-in-the-loop simulation)
- MAVProxy (MAVLink router) forwarding to UDP:14550
- Map and console windows for visualization

### 16.2 Running a Tracking Script

```bash
# Terminal 2: Activate venv and run
cd /path/to/EAGLE_EYE_X
source VENV/uav/bin/activate

# Test ASPS logic (no drone needed):
cd src/ASPS_LIDAR
python -c "
from prioritization import ASPSPrioritySelector
s = ASPSPrioritySelector()
best, comps = s.select(
    [{'cls':0,'conf':0.9,'xyxy':[0,0,50,50]},
     {'cls':2,'conf':0.7,'xyxy':[200,200,300,300]}],
    1000, {2:4,1:3,3:2,0:1}, lambda d: 2.0
)
print(best, comps)
"

# Full script (requires SITL running):
python ASPS_LIDAR_TR.py
```

### 16.3 Switching to GUIDED Mode

The script waits for `FlightMode.OFFBOARD` before starting tracking. In MAVProxy:
```
MAV> mode guided
```

Or through QGroundControl: switch flight mode to GUIDED.

### 16.4 Simulating a Target

Since no real camera is available in SITL, set `CAMERA_INDEX = 2` and point any USB camera at a screen with YOLO-detectable objects. Or use a video loopback:
```bash
# Play video file as camera
gst-launch-1.0 filesrc location=test_video.mp4 ! decodebin ! videoconvert ! v4l2sink device=/dev/video2
```

### 16.5 ArduPilot SITL Details

The connection chain:
```
ArduCopter SITL → TCP:5760 → MAVProxy → UDP:14550 ← MAVSDK (script)
                                         → TCP:5760 ← mavproxy console
```

Port mapping (instance 0):
- TCP 5760: SITL telemetry
- TCP 5501: SITL control
- UDP 14550: MAVProxy GCS output (what scripts connect to)

---

## 17. Ablation & Metrics

### 17.1 ASPS Ablation Design

The `USE_ASPS` toggle in `prioritization.py` enables direct comparison:

```
USE_ASPS = True   → ASPS scoring (weighted components + hysteresis)
USE_ASPS = False  → pure max(CLASS_PRIORITIES) — identical to lidar_trt.py
```

Run two flights (same conditions, same target path) with ASPS on/off and compare the CSV logs.

### 17.2 Metrics to Collect

| Metric | How | CSV columns used |
|---|---|---|
| Target switching frequency | `df['Class'].diff().abs().sum()` | `Class` |
| Mean tracking duration per target | Group by `Class`, mean of consecutive same-class runs | `Class`, `Time` |
| Number of unnecessary switches | Switch when `FinalP` difference < 0.05 | `Class`, `FinalP` |
| Mean priority score | `df['FinalP'].mean()` | `FinalP` |
| Time in FOLLOW vs CENTER | `df['Target_Centered'].mean()` | `Target_Centered` |
| Distance error | `df['Dist_Err_m'].mean()`, `.std()` | `Dist_Err_m` |
| Forward velocity distribution | `df['Fwd_Vel_mps'].describe()` | `Fwd_Vel_mps` |
| End-to-end latency impact | Compare overall FPS between modes | `KF_dt` |
| Component contributions | `df[['BaseP','ConfP','DistP','VelP']].mean()` | ASPS columns |

### 17.3 Analysis Script Template

```python
import pandas as pd
import matplotlib.pyplot as plt

df_asps_on = pd.read_csv('asps_on/log.csv')
df_asps_off = pd.read_csv('asps_off/log.csv')

# Switching frequency
for name, df in [('ASPS ON', df_asps_on), ('ASPS OFF', df_asps_off)]:
    switches = (df['Class'].diff() != 0).sum()
    duration = df['Time'].iloc[-1] - df['Time'].iloc[0]
    print(f"{name}: {switches} switches in {duration:.0f}s ({switches/duration:.2f}/s)")

# Component breakdown (ASPS ON only)
print("\nASPS Component Means:")
print(df_asps_on[['BaseP','ConfP','DistP','VelP','FinalP']].mean())

# Distance error
print("\nDistance Error (m):")
for name, df in [('ASPS ON', df_asps_on), ('ASPS OFF', df_asps_off)]:
    print(f"  {name}: mean={df['Dist_Err_m'].mean():.3f}, std={df['Dist_Err_m'].std():.3f}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(df_asps_on['Time'], df_asps_on['Dist_Err_m'], label='ASPS ON')
axes[0].plot(df_asps_off['Time'], df_asps_off['Dist_Err_m'], label='ASPS OFF')
axes[0].set_ylabel('Distance Error (m)')
axes[0].legend()

axes[1].plot(df_asps_on['Time'], df_asps_on['Class'], label='ASPS ON', alpha=0.7)
axes[1].plot(df_asps_off['Time'], df_asps_off['Class'], label='ASPS OFF', alpha=0.7)
axes[1].set_ylabel('Class ID')
axes[1].set_xlabel('Time (s)')
axes[1].legend()
plt.savefig('ablation_comparison.png')
```

---

## 18. Known Issues & Pitfalls

### 18.1 Current Issues

| Issue | Affected scripts | Status |
|---|---|---|
| **Altitude PID oscillates** | All | Logs show ±1.5m oscillation. Needs retuning: try `KP_ALT=0.3, KI_ALT=0.01, KD_ALT=0.15` |
| **Loop runs at ~1 FPS** | All | Likely YOLO running on CPU. Fix: ensure CUDA is available and use `.engine` (TensorRT) for inference. Check `torch.cuda.is_available()` |
| **SITL connection drops** | All | MavProxy gets "Closed connection on SERIAL0" intermittently. Workaround: restart SITL and mavproxy |
| **VelocityTracker ignores object ID** | ASPS scripts | Two targets of same class share velocity state. Fine for single-target tracking but could cause confusion |

### 18.2 Known Pitfalls

- **Camera index**: `CAMERA_INDEX` varies by system. `0` is usually built-in, `2` is often USB on Jetson. Check with `ls /dev/video*`
- **Frame copy**: Always `.copy()` before putting frame in display queue. Without it, the main thread redraws on the same frame being displayed
- **queue.Full**: `put_nowait` raises `queue.Full` if the queue is full. This is caught and ignored — the display thread was too slow, old frame is dropped
- **Display thread crash**: If the display thread crashes (e.g., OpenCV window closed externally), `display_should_stop` must be set to prevent the main loop from hanging
- **Mock LiDAR vs real LiDAR**: The mock returns a new distance every call (even without a target). In real mode, `read_distance()` returns `None` when no valid frame is available. Make sure your code handles `None` correctly
- **sys.path for imports**: `ASPS_LIDAR_TR.py` imports `from prioritization import ...`. This works when running from `src/ASPS_LIDAR/` because Python adds the script's directory to `sys.path`. Run with: `cd src/ASPS_LIDAR && python ASPS_LIDAR_TR.py`

### 18.3 Debugging Tips

```bash
# Check if CUDA is available for PyTorch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Check camera index
for i in /dev/video*; do echo $i; v4l2-ctl --list-formats -d $i 2>/dev/null; done

# Test LiDAR serial
python -c "import serial; s=serial.Serial('/dev/ttyUSB0',115200,timeout=1); print(s.read(20))"

# Watch CSV output in real-time
tail -f flights/asps_lidar_*/log_*.csv

# Check SITL connection
python -c "from pymavlink import mavutil; m=mavutil.mavlink_connection('udpin:127.0.0.1:14550'); m.wait_heartbeat(); print('Connected')"
```

---

## 19. File Reference

### 19.1 Complete File List

```
EAGLE_EYE_X/
├── agentdev.md                      ← THIS FILE
├── src/
│   ├── lastscr.py                   ← Original base (untouched)
│   ├── lastscr_trt.py               ← v2: per-frame dt + display thread + 4-class priorities
│   ├── lidar_trt.py                 ← LiDAR distance following
│   ├── ASPS_LIDAR/
│   │   ├── ASPS_LIDAR_TR.py         ← ASPS + LiDAR distance following
│   │   ├── prioritization.py        ← ASPS engine
│   │   ├── README.md                ← Usage guide
│   │   └── AGENTS.md                ← Agent instructions
│   ├── upg7.py                      ← Reference: ZoeDepth + YOLO depth follow
│   ├── upg9.py                      ← Reference: display thread pattern
│   ├── upg5repl.py                  ← Reference: 4-class priorities + raw TRT class
│   ├── lightscr.py                  ← Reference: per-frame dt KF pattern
│   └── wl.py                        ← Reference: LiDAR serial + mock pattern
├── MDE/                             ← Monocular depth estimation scripts
│   ├── zoe.py                       ← ZoeDepth standalone
│   ├── monocular.py                 ← MiDaS tracking loop
│   └── mdetest.py                   ← MiDaS standalone test
├── VENV/uav/                        ← Python venv with all dependencies
├── flights/                         ← Flight log output directory
└── .vscode/settings.json            ← VS Code Python interpreter config
```

### 19.2 Dependency Inventory

```
# Core (required for all scripts)
torch
ultralytics
opencv-python
mavsdk
numpy

# LiDAR (optional — mock mode works without it)
pyserial

# MDE depth (for depth_trt.py, not yet built)
transformers
Pillow
scipy
trl

# SITL testing
ardupilot (system install: sim_vehicle.py)
mavproxy (pip install mavproxy)
```

### 19.3 Script Dependency Graph

```
lastscr.py
    │
    ├──→ lastscr_trt.py  (KF dt fix + display thread + 4-class priorities)
    │         │
    │         ├──→ lidar_trt.py  (+ LiDAR + DistKF + DistPID + VelocityNedYaw)
    │         │       │
    │         │       └──→ ASPS_LIDAR/ASPS_LIDAR_TR.py  (+ ASPSPrioritySelector)
    │         │
    │         └──→ depth_trt.py  (future: + ZoeDepth/MiDaS + DistKF + DistPID)
    │
    └──→ (reference only: upg7.py, upg9.py, lightscr.py, upg5repl.py)
```

---

*End of agentdev.md — generated 2026-07-03*
