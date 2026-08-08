# Setup Guide

Two supported environments:

| Environment | Purpose | Stack |
|---|---|---|
| **PC / x86 Linux** | Development, SITL testing, model training, ablation | CPU/GPU PyTorch + MAVSDK, no hardware |
| **Jetson Orin Nano** | Onboard deployment | JetPack L4T PyTorch + TensorRT, camera + LiDAR + FC |

Python 3.10 is used throughout.

---

## 1. PC / SITL development setup

```bash
# 1. clone
git clone git@github.com:WOAH-KAMRAN/EAGLE_EYE_X.git
cd EAGLE_EYE_X

# 2. create + populate the virtualenv (same layout the team uses)
python -m venv --system-site-packages VENV/uav
source VENV/uav/bin/activate
pip install -r requirements.txt

# or, one command:
make setup

# 3. fetch development model weights into models/
make models
```

> **Important:** always run the tracking/model scripts **from the repo root**. Paths
> like `models/best.pt` and `flights/` are resolved relative to the current directory.

### Model weights

| Weight | Purpose | How to get |
|---|---|---|
| `models/yolov8n.pt` | SITL/dev base model (COCO, auto-usable for the 4 priority classes) | `make models` (auto-download) |
| `models/best.pt` | Custom 4-class model — **private until paper publication** | Request from a maintainer, drop into `models/` |
| `models/best.engine` | TensorRT engine for Jetson | `python model/export_trt.py --model models/best.pt --half` |

Verify:

```bash
source VENV/uav/bin/activate
python -c "import torch, ultralytics, mavsdk, cv2; print(torch.__version__, ultralytics.__version__)"
```

### SITL (no drone hardware)

1. Run ArduPilot SITL on your machine, reachable at `udpin://127.0.0.1:14550`.
2. Have a webcam available (`CAMERA_INDEX` in `configs/hardware.yaml`).
3. Launch the tracking pipeline in mock-LiDAR mode:

```bash
make sitl
# equivalent: python src/ASPS_LIDAR/ASPS_LIDAR_TR.py --config configs/sitl.example.yaml
```

The pipeline connects, takes off to `TARGET_ALTITUDE`, and starts detecting/prioritizing.
Switch the FC to `OFFBOARD`/`GUIDED` to engage tracking.

---

## 2. Jetson Orin Nano setup

The x86 requirements are **not** valid on the Jetson — NVIDIA ships L4T-specific
PyTorch wheels. Procedure:

1. Install **JetPack 5.x or 6.x** (SDK Manager). This provides CUDA, TensorRT, cuDNN,
   and an OpenCV build with GStreamer + CUDA (do **not** `pip install opencv-python` over it).
2. Create the venv **with system packages** so JetPack OpenCV/TensorRT are visible:

   ```bash
   python -m venv --system-site-packages VENV/uav
   source VENV/uav/bin/activate
   ```

3. Install PyTorch from the NVIDIA wheel index for your JetPack:

   ```bash
   # JetPack 6.x
   pip install torch torchvision --index-url https://download.pytorch.org/whl/nv-jp6x
   # JetPack 5.x
   pip install torch torchvision --index-url https://download.pytorch.org/whl/nv-jp5x
   ```

4. Install the rest:

   ```bash
   pip install -r requirements-jetson.txt
   ```

5. TensorRT Python bindings come from the JetPack SDK:

   ```bash
   sudo apt install python3-libnvinfer python3-libnvinfer-dev
   ```

6. Configure hardware in `configs/hardware.yaml` (camera index / CSI, `YOLO_MODEL_PATH`
   pointing at your exported `.engine`, LiDAR port) and run:

   ```bash
   python src/ASPS_LIDAR/ASPS_LIDAR_TR.py
   ```

---

## 3. Personal overrides (per-developer)

Copy any config file into `configs/local/` (gitignored) and edit — your overrides are
applied on top of the committed defaults and never conflict with teammates:

```bash
mkdir -p configs/local
cp configs/hardware.yaml configs/local/hardware.yaml   # edit as needed
```

Precedence (later wins): script defaults → `configs/*.yaml` → `configs/local/*.yaml`
→ `--config <path>`.

---

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| `PyYAML not installed` | `pip install pyyaml` |
| `pyserial not installed` | Install `pyserial` for real LiDAR; mock mode works without it |
| `CUDA not available` | Fine for SITL/CPU; `model/train.py --device cpu` |
| Model file not found | `YOLO_MODEL_PATH` is relative to CWD — run from repo root, or use an absolute path in `configs/local/hardware.yaml` |
| No webcam on SITL box | Adjust `CAMERA_INDEX` or point the detection at a video source |
