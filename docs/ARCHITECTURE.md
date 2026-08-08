# Architecture

```
              ┌────────────────────────────────────────────────────────────┐
              │                    JETSON ORIN NANO                         │
              │                                                            │
  Camera ────▶│  YOLO detector ─▶ ASPS priority selector ─▶ target box      │
 (CSI/USB)    │                     │  (class, conf, dist, vel)            │
              │                     ▼                                      │
              │              Kalman filter (yaw, pitch, dist)              │
              │                     │                                      │
              │                     ▼                                      │
              │        Twin PID loops ─── yaw/pitch centering              │
              │        Distance PID  ─── forward velocity (LiDAR/depth)    │
              │                     │                                      │
              │                     ▼                                      │
              │              MAVSDK ──▶ flight controller (ArduPilot)      │
              │                                                            │
 TF Mini ────▶│  LiDAR read (real or mock) ── DistanceKalmanFilter         │
              │                                                            │
              │  CSV logger (flight logs)  +  video recorder               │
              └────────────┬───────────────────────────────────────────────┘
                           │ MAVLink heartbeat (SCR_USER1)
                           ▼
              Lua failsafe (on FC) ── RTL / restart companion on timeout
```

## Modules

| Module | Responsibility |
|---|---|
| `src/ASPS_LIDAR/ASPS_LIDAR_TR.py` | Main loop: capture, telemetry, control, logging, display thread |
| `src/ASPS_LIDAR/prioritization.py` | **ASPS** scoring: `ALPHA*P_base + BETA*conf + GAMMA*dist + DELTA*vel` + hysteresis |
| `src/ASPS_LIDAR/config.py` | YAML config layering (`configs/`, `configs/local/`, `--config`) |
| `model/` | Training / evaluation / inference / TensorRT export |
| `tools/analyze_flight.py` | Offline flight-log metrics + figures |
| `experiments/ablation/` | SITL config-matrix runner |
| `experiments/depth/` | Monocular-depth experiments (MiDaS/ZoeDepth), calibration |
| `failsafe/failsafe.lua` | On-FC heartbeat + RTL failsafe |

## Key design points

1. **ASPS**: mission-defined class hierarchy is the dominant term (`ALPHA=0.60`);
   runtime signals (confidence, distance, velocity) refine it. `SWITCH_THRESHOLD`
   provides hysteresis against target churn. Toggling `USE_ASPS` yields the pure
   `max(CLASS_PRIORITIES)` baseline for ablations.
2. **Control**: two-stage — first center the target (yaw/pitch `AttitudeRate`),
   then once centered (`abs(yaw_err) <= YAW_CENTERED_THRESHOLD`) close the distance
   loop with forward velocity.
3. **Smoothing**: per-frame `dt` Kalman filters for centering (pixels) and distance (m).
4. **Safety**: RC-override detection, altitude limits, telemetry/detection timeouts,
   emergency land, Lua failsafe heartbeat.

## Flight log schema (current, ASPS)

```
Time, Mode, Alt, Alt_Err,
Yaw_Err, Pitch_Err,
Yaw_Cmd, Pitch_Cmd, Raw_LiDAR_m, KF_Dist_m,
Dist_Err_m, Fwd_Vel_mps, Target_Centered,
Class, Conf, RC_OK, KF_dt,
BaseP, ConfP, DistP, VelP, FinalP
```

Older logs from previous scripts use a reduced schema; `tools/analyze_flight.py`
handles both.

## Legacy code

All superseded iterations are preserved verbatim in `archive/` (grouped by original
location). See `archive/README.md` for the full lineage map.
