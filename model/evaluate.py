#!/usr/bin/env python3
"""Evaluate a trained YOLO model: mAP, precision, recall, confusion matrix.

Usage:
    python model/evaluate.py --model models/best.pt --data model/data.yaml
    python model/evaluate.py --model model/runs/train/weights/best.pt --data model/data.yaml

Outputs: model/runs/<name>/  (metrics.csv, confusion_matrix.png, PR/curves)
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


def main():
    ap = argparse.ArgumentParser(description="Evaluate YOLO on Eagle Eye X val split")
    ap.add_argument("--model", required=True, help="trained weights (best.pt)")
    ap.add_argument("--data", default="model/data.yaml", help="dataset YAML (template)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="eval", help="ultralytics run name")
    ap.add_argument("--project", default="model/runs")
    args = ap.parse_args()

    with open(args.data) as fh:
        cfg = _expand_env(yaml.safe_load(fh))
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="eagle_data_")
    with os.fdopen(fd, "w") as fh:
        yaml.safe_dump(cfg, fh)

    model = YOLO(args.model)
    metrics = model.val(
        data=tmp,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        name=args.name,
        project=args.project,
    )

    print("\n===== Evaluation summary =====")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print(f"Results saved to {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
