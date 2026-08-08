# ASPS LiDAR Tracking

Adaptive Semantic Priority Selection (ASPS) + TF Mini LiDAR for UAV target following.

## Architecture

```
YOLO detection ──→ ASPSPrioritySelector ──→ Best target (cyan box)
                      │
              ┌───────┼───────────┐
              │       │           │
         Base P.  Confidence  Distance  Velocity
         (0.60)    (0.20)     (0.15)    (0.05)
              │       │           │
        CLASS_PRIORITIES    TF Mini LiDAR  bbox px/track
```

## Files

| File | Purpose |
|---|---|
| `ASPS_LIDAR_TR.py` | Main tracking script (run this) |
| `prioritization.py` | ASPS scoring engine (config + classes) |

## Dependencies

Same as `lidar_trt.py`: torch, ultralytics, mavsdk, opencv-python, numpy, pyserial (optional for mock mode).

## Usage

```bash
source VENV/uav/bin/activate
python src/ASPS_LIDAR/ASPS_LIDAR_TR.py            # uses configs/*.yaml defaults
python src/ASPS_LIDAR/ASPS_LIDAR_TR.py --config experiments/ablation/configs/run.yaml   # experiment override
```

### Configuration

Constants are layered from YAML at startup (`src/ASPS_LIDAR/config.py`); any key you
omit falls back to the built-in default. Precedence (later wins):

1. built-in script defaults
2. `configs/hardware.yaml`, `configs/control.yaml`, `configs/asps.yaml`, `configs/logging.yaml`
3. `configs/local/*.yaml` (per-developer overrides, gitignored)
4. an explicit file passed with `--config <path>`

- **`configs/asps.yaml`** — ASPS weights (`ALPHA`–`DELTA`), `SWITCH_THRESHOLD`, `USE_ASPS` toggle, `CLASS_PRIORITIES`
- **`configs/control.yaml`** — PID gains, Kalman Q/R, limits, follow distance
- **`configs/hardware.yaml`** — camera index, LiDAR port, `CONNECTION_STRING`, model path
- **`configs/logging.yaml`** — video recording, flight-data folder, timeouts

### ASPS toggle

Set `USE_ASPS = False` in `prioritization.py` to fall back to pure `CLASS_PRIORITIES` selection (ablation baseline).

## CSV Logging

Extended with 5 ASPS columns:

```
BaseP, ConfP, DistP, VelP, FinalP
```

Use these for ablation studies and paper figures.

## Control Flow

```
detect → center yaw/pitch (AttitudeRate)
  └── if yaw_err <= 30px ──→ read LiDAR → DistKF → DistancePID → VelocityNedYaw
```

ASPS runs every frame regardless of centering state.
