#!/usr/bin/env python3
"""Run YOLO inference on images, a video, or the webcam.

Replaces the old modelimg.py / modelrun.py root scripts.

Usage:
    python model/predict.py --model models/best.pt --source images/    # all jpg/png
    python model/predict.py --model models/best.pt --source 0          # webcam
    python model/predict.py --model models/best.pt --source clip.mp4
    python model/predict.py --model models/best.pt --source image.jpg --save

Output: annotated frames saved to results/ (or --outdir).
"""
import argparse
import glob
import os

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="YOLO inference on images/video/webcam")
    ap.add_argument("--model", default="models/best.pt", help="trained weights")
    ap.add_argument("--source", required=True, help="image, image dir, video path, or camera index (0)")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--show", action="store_true", help="display annotated frames live")
    ap.add_argument("--save", action="store_true", help="save annotated outputs")
    args = ap.parse_args()

    model = YOLO(args.model)
    os.makedirs(args.outdir, exist_ok=True)

    sources = []
    if os.path.isdir(args.source):
        sources = sorted(glob.glob(os.path.join(args.source, "*.jpg")) +
                         glob.glob(os.path.join(args.source, "*.png")))
    elif os.path.exists(args.source):
        sources = [args.source]
    elif args.source.isdigit():
        sources = [int(args.source)]
    else:
        raise SystemExit(f"Source not found: {args.source}")

    for src in sources:
        results = model.predict(src, conf=args.conf, verbose=False)
        for r in results:
            frame = r.orig_img.copy()
            if r.boxes is not None:
                frame = r.plot()
            if args.show:
                cv2.imshow("Eagle Eye X", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise SystemExit
            if args.save:
                name = os.path.basename(src) if isinstance(src, str) else "webcam.jpg"
                out = os.path.join(args.outdir, name)
                cv2.imwrite(out, frame)
                print(f"saved {out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
