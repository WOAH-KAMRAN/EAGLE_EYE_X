# Eagle Eye X — developer workflow
# See docs/SETUP.md for environment setup on both PC (SITL) and Jetson.

VENV  := VENV/uav
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip
DATA  ?= dataset            # dataset root passed to model/*.py (override with DATA=...)

.PHONY: help setup setup-jetson models sitl train eval predict trt analyze ablation replay clean

help:
	@echo "Eagle Eye X targets:"
	@echo "  make setup           create venv + install requirements.txt (PC/SITL)"
	@echo "  make setup-jetson    print Jetson JetPack install steps"
	@echo "  make models          fetch dev YOLO weights into models/ (best.pt stays private)"
	@echo "  make sitl            run tracking against SITL (mock LiDAR, no hardware)"
	@echo "  make train           train YOLO (DATA=/path/to/dataset)"
	@echo "  make eval            evaluate a trained model (DATA=/path/to/dataset)"
	@echo "  make predict         run inference on images/video/webcam"
	@echo "  make trt             export .pt -> TensorRT .engine"
	@echo "  make analyze         compute metrics/plots from a flight CSV (F=<log.csv>)"
	@echo "  make ablation        run the SITL ablation matrix"
	@echo "  make replay          offline ASPS vs baseline replay on a video (V=<clip> [GT=<gt.csv>])"
	@echo "  make clean           remove __pycache__ and egg-info"

setup:
	python -m venv --system-site-packages $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

setup-jetson:
	@echo "Jetson setup is interactive — follow docs/SETUP.md (requires JetPack SDK)."

models:
	@mkdir -p models
	@if [ -f models/yolov8n.pt ]; then \
		echo "models/yolov8n.pt already present."; \
	else \
		echo "Fetching base YOLO dev weights (yolov8n.pt)..."; \
		$(PY) -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('downloaded')"; \
		mv -f yolov8n.pt models/yolov8n.pt; \
	fi
	@echo ""
	@echo "Done. models/yolov8n.pt ready for SITL / development."
	@echo "NOTE: best.pt (custom 4-class model, private until paper) is NOT in the repo."
	@echo "      Obtain it from a maintainer and drop it into models/best.pt."

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

replay:
	$(PY) experiments/replay/replay_asps.py --source $(V) $(if $(GT),--gt $(GT),)
