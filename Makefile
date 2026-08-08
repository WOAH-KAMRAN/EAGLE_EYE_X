# Eagle Eye X — developer workflow
# See docs/SETUP.md for environment setup on both PC (SITL) and Jetson.

VENV  := VENV/uav
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip
DATA  ?= dataset            # dataset root passed to model/*.py (override with DATA=...)

.PHONY: help setup setup-jetson sitl train eval predict trt analyze ablation clean

help:
	@echo "Eagle Eye X targets:"
	@echo "  make setup           create venv + install requirements.txt (PC/SITL)"
	@echo "  make setup-jetson    print Jetson JetPack install steps"
	@echo "  make sitl            run tracking against SITL (mock LiDAR, no hardware)"
	@echo "  make train           train YOLO (DATA=/path/to/dataset)"
	@echo "  make eval            evaluate a trained model (DATA=/path/to/dataset)"
	@echo "  make predict         run inference on images/video/webcam"
	@echo "  make trt             export .pt -> TensorRT .engine"
	@echo "  make analyze         compute metrics/plots from a flight CSV (F=<log.csv>)"
	@echo "  make ablation        run the SITL ablation matrix"
	@echo "  make clean           remove __pycache__ and egg-info"

setup:
	python -m venv --system-site-packages $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

setup-jetson:
	@echo "Jetson setup is interactive — follow docs/SETUP.md (requires JetPack SDK)."

sitl:
	@echo "Connecting to SITL at udpin://127.0.0.1:14550 (mock LiDAR)."
	$(PY) src/ASPS_LIDAR/ASPS_LIDAR_TR.py --config configs/sitl.example.yaml

train:
	$(PY) model/train.py --data "$(DATA)"

eval:
	$(PY) model/evaluate.py --data "$(DATA)"

predict:
	$(PY) model/predict.py

trt:
	$(PY) model/export_trt.py

analyze:
	$(PY) tools/analyze_flight.py "$(F)"

ablation:
	$(PY) experiments/ablation/run_ablation.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
