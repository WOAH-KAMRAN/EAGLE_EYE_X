# Eagle Eye X
A Target Prioritization & Tracking System for UAVs using NVIDIA JETSON ORIN NANO
# Project Overview

Eagle Eye X is a physical AI system designed for multirotor UAVs, integrating object detection, monocular depth estimation, and autonomous control into a single pipeline running on the NVIDIA Jetson Orin Nano.

It allows UAVs to detect, prioritize, and track high-value targets in real time using a single RGB camera, a custom priority based system, and control algorithms.\

# ⚠️ Research Notice

This repository accompanies an ongoing research internship project, with a corresponding paper under review for publication.
Certain resources, including trained AI models, datasets, and detailed flight logs, are not publicly included.
This repo demonstrates system architecture, integration, and workflow without revealing proprietary research data.
Full release will follow after the paper’s acceptance.

**For collaboration or academic inquiries, please contact the maintainer directly.**


# System Overview

Eagle Eye X consists of four main components and an additional telemetry module in development:

**Description**

**1. Core Python Script**: Runs on the Jetson Orin Nano; integrates object detection, MiDaS depth estimation, Kalman filter smoothing, twin PID loops, and MAVSDK based control  all in one script.

**2. Ground Control Station (GCS)**:	Custom interface for monitoring system health, telemetry, and video feed, with mission control capabilities.

**3. Lua Failsafe Script**:	Runs inside the flight controller to handle communication loss, emergency scenarios, or target loss.

**4. Drone Bridge (ESP32 Telemetry)**:	Under development; provides low-latency UAV to GCS telemetry link.

**5. UAV  testbed**: integrating Jetson, camera, flight controller, and telemetry hardware 


# CORE TECH
Single-camera depth estimation using MiDaS (monocular depth)

Custom target prioritization based on class hierarchy

Kalman filter smoothing for stable tracking

Twin PID control loops for yaw and pitch alignment

Twin-dot target alignment system for visual confirmation

Ground Control Station (GCS) with telemetry, video overlay, and control toggles

Failsafe Lua script for emergency handling

ESP32 Drone Bridge (under development) for telemetry communication


# License

Released under the MIT License.
Some components (trained models/dataset) will remain private until paper publication.

# Maintainers

1. Kamran Ahmad [@WOAH-KAMRAN]
2. Melvin Joseph [@AVATAR3905]
4. Ayushya Ranjan [@Dagger7164]

# Mentor
1. Dr. Sachin Gupta [@Sachinmait]
