# Eagle Eye X

A Target Prioritization & Tracking System for UAVs using NVIDIA JETSON ORIN NANO.

Eagle Eye X is a physical AI system for multirotor UAVs: real-time object detection,
custom **Adaptive Semantic Priority Selection (ASPS)**, Kalman-smoothed tracking, and
autonomous control in a single pipeline running on a Jetson Orin Nano. It detects,
prioritizes, and follows high-value targets with one RGB camera (plus optional TF Mini
LiDAR distance following).

## Quickstart

```bash
git clone git@github.com:WOAH-KAMRAN/EAGLE_EYE_X.git && cd EAGLE_EYE_X
make setup                       # venv + deps (PC/SITL)
make sitl                        # run against ArduPilot SITL (mock LiDAR)
```

See [docs/SETUP.md](docs/SETUP.md) for PC and Jetson setup, [docs/HARDWARE.md](docs/HARDWARE.md)
for wiring, and [docs/MODEL_TESTING.md](docs/MODEL_TESTING.md) for training/evaluation.

## Repository layout

```
src/ASPS_LIDAR/          production mainline (tracking pipeline + ASPS + config loader)
configs/                 YAML configuration (asps / control / hardware / logging)
model/                   training, evaluation, inference, TensorRT export
tools/analyze_flight.py  flight-log metrics + paper figures
experiments/
  ├── depth/             monocular-depth experiments (MiDaS / ZoeDepth) + calibration
  └── ablation/          SITL ablation matrix runner
failsafe/                Lua heartbeat + RTL failsafe for the flight controller
docs/                    setup, hardware, architecture, model testing, research protocol
archive/                 ALL legacy scripts, preserved verbatim (see archive/README.md)
```

## System overview

| Component | Description |
|---|---|
| Core pipeline | `src/ASPS_LIDAR/ASPS_LIDAR_TR.py` — detection, ASPS prioritization, Kalman filtering, twin PID (yaw/pitch), distance PID, MAVSDK control, CSV + video logging |
| ASPS | `prioritization.py` — `P_final = α·P_base + β·conf + γ·dist + δ·vel` with hysteresis |
| Ground Control Station | custom monitoring interface (telemetry/video overlay, mission control) |
| Failsafe | `failsafe/failsafe.lua` on the flight controller (heartbeat → RTL) |
| Drone Bridge | ESP32 telemetry link — under development |

**Core tech:** MiDaS monocular depth · ASPS target prioritization · Kalman smoothing ·
twin PID loops · twin-dot target alignment · Lua failsafe · TensorRT YOLO.

## Research

This repository accompanies an ongoing research internship; a paper is under review.

- **Research notice:** trained models, datasets, and detailed flight logs are private
  until paper acceptance. This repo demonstrates architecture, integration, and workflow.
- **Methodology & metrics:** see [docs/RESEARCH.md](docs/RESEARCH.md) (ablation protocol,
  metric definitions, figure recipes).
- **Ablation harness:** `experiments/ablation/run_ablation.py` runs the ASPS vs
  baseline matrix over SITL and produces comparison tables.

For collaboration or academic inquiries, contact the maintainers.

## License

MIT — see [LICENSE](LICENSE). Trained models/dataset remain private until publication.

## Maintainers

1. Kamran Ahmad [@WOAH-KAMRAN]
2. Melvin Joseph [@AVATAR3905]
3. Ayushya Ranjan [@AyushyaRanjan]

**Mentor:** Dr. Sachin Gupta [@Sachinmait]
