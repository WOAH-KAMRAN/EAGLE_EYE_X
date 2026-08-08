#!/usr/bin/env python3
"""Train a YOLO model on the Eagle Eye X dataset.

Usage:
    python model/train.py --model yolov8n.pt --data model/data.yaml --epochs 100
    python model/train.py --model models/best.pt --data model/data.yaml   # fine-tune

Env: $EAGLE_DATASET is substituted into ${...} placeholders in the data YAML.
Outputs go to model/runs/<name> (gitignored).
"""
import argparse
import os
import tempfile

import yaml
from ultralytics import YOLO


def _expand_env(value):
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def materialize_data_yaml(data_path):
    """Render ${ENV_VAR} placeholders into a usable temp data YAML."""
    with open(data_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg = _expand_env(cfg)
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="eagle_data_")
    with os.fdopen(fd, "w") as fh:
        yaml.safe_dump(cfg, fh)
    print(f"Rendered dataset config -> {tmp}")
    print(f"  dataset path: {cfg.get('path')}")
    return tmp


def main():
    ap = argparse.ArgumentParser(description="Train YOLO on Eagle Eye X dataset")
    ap.add_argument("--data", default="model/data.yaml", help="dataset YAML (template)")
    ap.add_argument("--model", default="yolov8n.pt", help="base weights (yolov8n.pt or models/best.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto-detect")
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu, or None=auto")
    ap.add_argument("--name", default="train", help="ultralytics run name")
    ap.add_argument("--project", default="model/runs")
    args = ap.parse_args()

    data = materialize_data_yaml(args.data)
    model = YOLO(args.model)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project=args.project,
    )
    print(f"Training complete. Weights: {args.project}/{args.name}/weights/")


if __name__ == "__main__":
    main()
