# Model Testing Guide

Eagle Eye X uses a custom-trained YOLO detector (4 classes). The **trained weights
and dataset are private** until the paper is published; this guide shows the team
workflow that works against the private data on your own machines.

## Class mapping (single source of truth)

`model/classes.yaml` defines the 4 classes:

```
0: person      1: bicycle      2: car      3: motorcycle
```

Keep it in sync with `CLASS_PRIORITIES` in `configs/asps.yaml` (class id → mission
priority). Training and the tracker must never disagree on ids.

## Dataset

Your dataset lives **outside the repo** (private). Expected layout:

```
<dataset>/images/{train,val,test}/<name>.jpg
<dataset>/labels/{train,val,test}/<name>.txt     # YOLO-format labels
```

Point the environment at it:

```bash
export EAGLE_DATASET=/path/to/eagle_dataset
```

`model/data.yaml` is a template; `model/train.py` substitutes `${EAGLE_DATASET}` at
runtime. (`model/dataset/` in the repo is a gitignored scratch area if you need one.)

## Training

```bash
source VENV/uav/bin/activate

# fresh nano model
make train DATA=/path/to/eagle_dataset
# or explicitly:
python model/train.py --model yolov8n.pt --data model/data.yaml --epochs 100 --imgsz 640

# fine-tune the existing custom weights
python model/train.py --model models/best.pt --data model/data.yaml --epochs 50

# GPU / CPU
python model/train.py --device 0        # GPU
python model/train.py --device cpu      # CPU
```

Outputs go to `model/runs/<name>/weights/` (gitignored).

## Evaluation

```bash
make eval        # uses models/best.pt + model/data.yaml
python model/evaluate.py --model model/runs/train/weights/best.pt --data model/data.yaml
```

Reports `mAP@0.5`, `mAP@0.5:0.95`, precision, recall, and saves the confusion
matrix + curves to `model/runs/<name>/`. Record these numbers for the model card
in the paper.

## Inference (detector sanity check)

```bash
python model/predict.py --model models/best.pt --source images/ --save
python model/predict.py --model models/best.pt --source 0 --show       # webcam
python model/predict.py --model models/best.pt --source clip.mp4 --save
```

Outputs land in `results/`.

## TensorRT export (Jetson deployment)

On the Orin (TensorRT required):

```bash
python model/export_trt.py --model models/best.pt --half
# -> models/best.engine
```

Then in `configs/hardware.yaml`:

```yaml
YOLO_MODEL_PATH: 'models/best.engine'
```

## Model card checklist

- [ ] weights path + git hash / dataset version recorded
- [ ] mAP@0.5, mAP@0.5:0.95, P, R captured
- [ ] inference FPS on the target device (Jetson)
- [ ] per-class confusion matrix reviewed
- [ ] `CLASS_PRIORITIES` matches `model/classes.yaml`
