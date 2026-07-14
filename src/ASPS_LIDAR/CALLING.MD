# Agents — ASPS LiDAR Tracking

## Context

This folder contains an Adaptive Semantic Priority Selection (ASPS) implementation integrated with TF Mini LiDAR-based distance following for UAV target tracking. The system runs on a Jetson Orin Nano with ArduPilot SITL or real drone hardware.

## Key Files

- `ASPS_LIDAR_TR.py` — main entry point, ~940 lines. Single-threaded async main loop + display thread.
- `prioritization.py` — ASPS module: `ASPSPrioritySelector`, `VelocityTracker`, `compute_priority()`, config constants.

## Architecture Notes

1. **ASPS weights**: `ALPHA=0.60, BETA=0.20, GAMMA=0.15, DELTA=0.05` — the mission-defined class hierarchy (`ALPHA * P_base`) remains dominant.

2. **Hysteresis**: `SWITCH_THRESHOLD = 0.15` — prevents rapid target switching. Only switches when `new_score > current_score + 0.15`.

3. **Velocity tracking**: `VelocityTracker` is keyed by `class_id`, not by persistent object ID. Two objects of the same class will share velocity state.

4. **Testing**: Set `USE_REAL_LIDAR = False` in `prioritization.py` for mock LiDAR mode. Works in SITL with no hardware.

5. **Ablation**: Set `USE_ASPS = False` — falls back to pure `max(CLASS_PRIORITIES)` selection. Run the same flight in both modes and compare the 5 ASPS CSV columns.

## Common Tasks

### Tune ASPS weights
Edit `prioritization.py` top section. Ensure `ALPHA + BETA + GAMMA + DELTA = 1.0`.

### Add a new priority component
1. Add weight constant in `prioritization.py`
2. Add score computation in `compute_priority()`
3. Update the `components` dict return value
4. Add CSV column in `ASPS_LIDAR_TR.py` header + log write
5. Add UI overlay display

### Port ASPS to another script (e.g., depth_trt.py)
```python
from prioritization import ASPSPrioritySelector

selector = ASPSPrioritySelector()
best_det, components = selector.select(
    detections, time.time(), CLASS_PRIORITIES,
    lambda det: depth_value_from_mde(frame, det['xyxy'])
)
```

## Pitfalls

- `selector.select()` expects `detections` to be a list of dicts with `'cls'`, `'conf'`, `'xyxy'` keys
- `distance_fn` is called once per detection per frame — keep it lightweight
- The `import serial` block at the top of the main script uses a try/except; mock mode does not require pyserial
- Reset `selector.reset()` when flight mode changes to OFFBOARD (handled in STANDBY branch)
