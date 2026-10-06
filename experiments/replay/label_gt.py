#!/usr/bin/env python3
"""Click-to-label ground truth for replay_asps.py.

Shows every Nth frame of the clip with the cached detections (from a replay run's
tracks.json). Click inside the box of the vehicle the UAV SHOULD follow.
Keys:  n = no valid target on this frame   s = skip frame   b = back   q = save + quit

Writes CSV `frame,x1,y1,x2,y2` (empty box = no target) for --gt.

Usage:
  python experiments/replay/label_gt.py --source clip.mp4 \
      --tracks experiments/replay/results/<run>/tracks.json --every 5 --out clip_gt.csv
"""
import argparse
import csv
import json
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the same video/folder used for the replay run")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = json.load(open(args.tracks))
    dets = {fr["frame"]: fr["dets"] for fr in cache["frames"]}
    if os.path.isdir(args.source):
        files = sorted(f for f in os.listdir(args.source) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        read = lambda i: cv2.imread(os.path.join(args.source, files[i]))  # noqa: E731
    else:
        cap = cv2.VideoCapture(args.source)

        def read(i):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            return img if ok else None

    labels = {}
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out)):
            labels[int(r["frame"])] = None if r["x1"] == "" else [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
    order = [f for f in sorted(dets) if f % args.every == 0]
    click = {}
    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("label", lambda e, x, y, *_: click.update(pt=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)

    i = 0
    while 0 <= i < len(order):
        f = order[i]
        img = read(f)
        if img is None:
            i += 1
            continue
        for d in dets[f]:
            x1, y1, x2, y2 = map(int, d["xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, f"id{d['id']} c{d['cls']} {d['conf']:.2f}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f"frame {f}  ({i + 1}/{len(order)})  label={labels.get(f, '-')}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("label", img)
        click.clear()
        while True:
            k = cv2.waitKey(30) & 0xFF
            if "pt" in click:
                x, y = click["pt"]
                hit = [d for d in dets[f] if d["xyxy"][0] <= x <= d["xyxy"][2] and d["xyxy"][1] <= y <= d["xyxy"][3]]
                if hit:   # smallest box under the click
                    d = min(hit, key=lambda d: (d["xyxy"][2] - d["xyxy"][0]) * (d["xyxy"][3] - d["xyxy"][1]))
                    labels[f] = d["xyxy"]
                    i += 1
                    break
                click.clear()
            if k == ord("n"):
                labels[f] = None
                i += 1
                break
            if k == ord("s"):
                i += 1
                break
            if k == ord("b"):
                i -= 1
                break
            if k == ord("q"):
                i = len(order)
                break
    cv2.destroyAllWindows()
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "x1", "y1", "x2", "y2"])
        for f in sorted(labels):
            w.writerow([f] + ([round(v, 2) for v in labels[f]] if labels[f] else ["", "", "", ""]))
    print(f"Saved {len(labels)} labels to {args.out}")


if __name__ == "__main__":
    main()
