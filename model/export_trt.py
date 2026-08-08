#!/usr/bin/env python3
"""Export a trained YOLO .pt model to TensorRT .engine for the Jetson Orin Nano.

Usage:
    python model/export_trt.py --model models/best.pt --half

Requires TensorRT (Jetson JetPack SDK). On a non-Jetson x86 box without TensorRT,
this will fail with a clear message — use the Orin or a TensorRT-enabled container.

Output: <model_dir>/<name>.engine  (e.g. models/best.engine)
"""
import argparse
import os

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Export YOLO .pt -> TensorRT .engine")
    ap.add_argument("--model", default="models/best.pt", help="trained weights")
    ap.add_argument("--half", action="store_true", help="FP16 engine (Jetson)"
                    "(ignored if unsupported)")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        raise SystemExit(
            "TensorRT not available here. Run this on the Jetson (JetPack SDK) "
            "or a TensorRT-enabled container."
        )

    model = YOLO(args.model)
    out = model.export(format="engine", imgsz=args.imgsz, half=args.half)
    print(f"\nTensorRT engine exported: {out}")
    print("Point the tracking script at it with:")
    print(f"  YOLO_MODEL_PATH: '{out}'  # in configs/hardware.yaml or --config")


if __name__ == "__main__":
    main()
