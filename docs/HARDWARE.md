# Hardware Guide

Testbed integration for the Eagle Eye X UAV.

## Bill of materials

| Item | Role | Notes |
|---|---|---|
| NVIDIA Jetson Orin Nano | Companion computer | Runs the tracking pipeline |
| RGB camera | Vision | RPi Cam v2 (CSI) or USB webcam |
| TF Mini (LiDAR) | Distance to target | USB-UART serial, 115200 baud |
| ArduPilot flight controller | Vehicle control | MAVSDK + MAVLink over UDP/serial |
| ESP32 (Drone Bridge) | Telemetry link | Under development |
| Multirotor airframe | Vehicle | — |

## Wiring / connections

- **Camera**: CSI connector on the Orin (RPi Cam v2) or USB. Set `CAMERA_INDEX` /
  resolution in `configs/hardware.yaml`.
- **TF Mini LiDAR**: USB-UART adapter → Orin. Default port `/dev/ttyUSB0`,
  baud 115200. Ground/5V/VCC/UTX/URX per the TF Mini datasheet. For testing without
  the sensor, keep `USE_REAL_LIDAR: False` (mock mode).
- **Flight controller ↔ Orin**: MAVSDK over UDP (`udpin://127.0.0.1:14550` for SITL;
  on the drone use the serial-to-UDP bridge, e.g. `mavlink-router` or a Telem2/`pymavlink`
  forwarder).
- **Failsafe heartbeat**: the Lua failsafe script (`failsafe/failsafe.lua`) runs on the
  flight controller and watches a heartbeat parameter (`SCR_USER1`) written by the
  companion Python loop. On timeout it triggers RTL.

## Pre-flight checklist

1. FC disarmed, RC bound and confirmed by the script (`RC: OK` overlay).
2. `CONNECTION_STRING` reachable (SITL or drone).
3. Camera produces frames (`CAMERA_INDEX` correct).
4. GPS lock and a safe takeoff area.
5. `TARGET_ALTITUDE` within `MIN_ALTITUDE`–`MAX_ALTITUDE`.
6. Emergency stop path known (Q / Ctrl+C lands and disarms).
7. Real LiDAR: check `/dev/ttyUSB0` permissions (`sudo usermod -aG dialout $USER`).

## Drone Bridge (ESP32) — status

Under development. Intended to provide a low-latency UAV→GCS telemetry link
(MAVLink-adjacent custom protocol). Not yet part of the main tracking pipeline.
