#!/usr/bin/env python3
"""Offline ASPS replay: compare target-selection policies on recorded video.

No drone, SITL or LiDAR needed. The detector runs ONCE per frame; detections are
associated into persistent tracks (IoU tracker) and cached, then every selection
policy is replayed over the same cached tracks so the comparison is fair.

Policies (see POLICIES below):
  asps_default        ASPS weights from configs/asps.yaml (0.60/0.20/0.15/0.05), hysteresis 0.15
  asps_balanced       0.25/0.25/0.25/0.25
  asps_velocity_boost 0.45/0.20/0.15/0.20
  asps_no_hysteresis  default weights, SWITCH_THRESHOLD = 0
  asps_hold           default ASPS + keep the current track through dropouts of <= HOLD_FRAMES
  class_priority      pure max(CLASS_PRIORITIES), ties -> confidence (current USE_ASPS=False)
  class_priority_hold class_priority + the same dropout hold (isolates what hold alone buys)
  nearest             largest box (closest by the distance proxy)
  max_confidence      highest detector confidence
  center_closest      box closest to the image centre

Unlike the flight script (which keys ASPS state by class id), this replay keys
velocity and hysteresis by TRACK id, so two vehicles of the same class are distinct.

Distance is a PROXY from box height: d = FOCAL_PX * OBJ_HEIGHT_M / box_h_px.
State this as a proxy in the paper.

Ground truth (optional, --gt): CSV with header `frame,x1,y1,x2,y2` giving the box of
the target that SHOULD be followed on that frame (label every Nth frame is fine;
unlabelled frames are skipped). A selection is correct if IoU(selected, gt) >= 0.5.
A row with empty box fields means "no valid target" (correct = select nothing).

Usage:
  python experiments/replay/replay_asps.py --source clip.mp4 --model models/best.pt
  python experiments/replay/replay_asps.py --source clip.mp4 --model models/best.engine --gt clip_gt.csv
  python experiments/replay/replay_asps.py --source images/ --fps 30          # folder of frames
  python experiments/replay/replay_asps.py --cache experiments/replay/results/<run>/tracks.json  # re-score only

Outputs (experiments/replay/results/<name>_<timestamp>/):
  tracks.json         cached detections + track ids + per-frame inference time
  sel_<policy>.csv    per-frame selection, score components, raw/KF centre
  summary.csv         one row per policy (paper table input)
  run_metadata.yaml   args, git rev, model, frames, timing
"""
import argparse
import csv
import datetime
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "ASPS_LIDAR"))

from config import load_flat_config  # noqa: E402

_cfg = load_flat_config()
CLASS_PRIORITIES = {int(k): v for k, v in
                    (_cfg.get("CLASS_PRIORITIES") or {2: 4, 1: 3, 3: 2, 0: 1}).items()}
ASPS_DEFAULT = dict(ALPHA=_cfg.get("ALPHA", 0.60), BETA=_cfg.get("BETA", 0.20),
                    GAMMA=_cfg.get("GAMMA", 0.15), DELTA=_cfg.get("DELTA", 0.05),
                    SWITCH_THRESHOLD=_cfg.get("SWITCH_THRESHOLD", 0.15))
Q_KF = _cfg.get("Q_KF", 0.1)
R_KF = _cfg.get("R_KF", 10)

POLICIES = {
    "asps_default": dict(kind="asps", **ASPS_DEFAULT),
    "asps_balanced": dict(kind="asps", ALPHA=0.25, BETA=0.25, GAMMA=0.25, DELTA=0.25,
                          SWITCH_THRESHOLD=ASPS_DEFAULT["SWITCH_THRESHOLD"]),
    "asps_velocity_boost": dict(kind="asps", ALPHA=0.45, BETA=0.20, GAMMA=0.15, DELTA=0.20,
                                SWITCH_THRESHOLD=ASPS_DEFAULT["SWITCH_THRESHOLD"]),
    "asps_no_hysteresis": dict(kind="asps", **{**ASPS_DEFAULT, "SWITCH_THRESHOLD": 0.0}),
    # ASPS + target persistence: keep the current track through short detection dropouts
    "asps_hold": dict(kind="asps", HOLD_FRAMES=10, **ASPS_DEFAULT),
    "class_priority": dict(kind="class_priority"),
    "class_priority_hold": dict(kind="class_priority", HOLD_FRAMES=10),   # fair control for asps_hold
    "nearest": dict(kind="nearest"),
    "max_confidence": dict(kind="max_confidence"),
    "center_closest": dict(kind="center_closest"),
}


# --- Frame sources -----------------------------------------------------------
def iter_frames(source):
    import cv2
    if os.path.isdir(source):
        files = sorted(f for f in glob.glob(os.path.join(source, "*"))
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                yield img
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {source}")
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield img
        cap.release()


def source_fps(source, default):
    if os.path.isdir(source):
        return default
    import cv2
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or default
    cap.release()
    return fps if fps > 1 else default


# --- Simple IoU tracker (dependency-free; ByteTrack-style greedy-Hungarian) ---
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class IoUTracker:
    """Associates detections frame-to-frame by IoU (Hungarian); tracks survive
    `max_age` missed frames. Class must match for association."""

    def __init__(self, iou_thr=0.3, max_age=15):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.tracks = {}   # id -> {'xyxy', 'cls', 'age'}
        self.next_id = 1

    def update(self, dets):
        from scipy.optimize import linear_sum_assignment
        ids = list(self.tracks)
        assigned = [None] * len(dets)
        if ids and dets:
            cost = np.ones((len(ids), len(dets)))
            for i, tid in enumerate(ids):
                for j, d in enumerate(dets):
                    if d["cls"] == self.tracks[tid]["cls"]:
                        cost[i, j] = 1.0 - iou(self.tracks[tid]["xyxy"], d["xyxy"])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= 1.0 - self.iou_thr:
                    assigned[c] = ids[r]
        matched = set(a for a in assigned if a is not None)
        for tid in ids:
            if tid not in matched:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        for j, d in enumerate(dets):
            tid = assigned[j]
            if tid is None:
                tid = self.next_id
                self.next_id += 1
            self.tracks[tid] = {"xyxy": d["xyxy"], "cls": d["cls"], "age": 0}
            d["id"] = tid
        return dets


# --- Detection pass ----------------------------------------------------------
def run_detection(args):
    from ultralytics import YOLO
    model = YOLO(args.model, task="detect")
    tracker = IoUTracker(iou_thr=args.track_iou, max_age=args.track_max_age)
    frames, h, w = [], None, None
    for idx, img in enumerate(iter_frames(args.source)):
        if args.max_frames and idx >= args.max_frames:
            break
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        res = model.predict(img, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                            half=args.half, device=args.device, verbose=False)[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0
        dets = []
        if res.boxes is not None:
            for b in range(len(res.boxes)):
                x1, y1, x2, y2 = (float(v) for v in res.boxes.xyxy[b].cpu().numpy())
                dets.append({"xyxy": [x1, y1, x2, y2], "cls": int(res.boxes.cls[b]),
                             "conf": float(res.boxes.conf[b])})
        dets = tracker.update(dets)
        frames.append({"frame": idx, "infer_ms": infer_ms, "dets": dets})
        if idx % 100 == 0:
            print(f"  frame {idx}: {len(dets)} dets, {infer_ms:.1f} ms")
    return {"width": w, "height": h, "model": args.model,
            "names": {int(k): v for k, v in model.names.items()}, "frames": frames}


# --- Selection policies ------------------------------------------------------
class Policy:
    def __init__(self, name, spec, width, height, fps, focal_px, obj_h_m):
        self.name, self.spec = name, spec
        self.cx, self.cy = width / 2.0, height / 2.0
        self.fps, self.focal_px, self.obj_h_m = fps, focal_px, obj_h_m
        self.current_id = None
        self.current_score = 0.0
        self.vel = {}      # track id -> (cx, cy, filtered px/s)
        self.last_best = None
        self.missed = 0

    def proxy_distance(self, d):
        bh = max(d["xyxy"][3] - d["xyxy"][1], 1.0)
        return self.focal_px * self.obj_h_m / bh

    def _update_velocity(self, dets):
        dt = 1.0 / self.fps
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            prev = self.vel.get(d["id"])
            if prev is None:
                v = 0.0
            else:
                raw = math.hypot(cx - prev[0], cy - prev[1]) / dt
                v = 0.8 * prev[2] + 0.2 * raw          # same smoothing as VelocityTracker
            self.vel[d["id"]] = (cx, cy, v)

    def _asps_score(self, d):
        s = self.spec
        base = CLASS_PRIORITIES.get(d["cls"], 0) / max(CLASS_PRIORITIES.values())
        conf = max(0.0, min(d["conf"], 1.0))
        dist = min(1.0, 1.0 / max(self.proxy_distance(d), 1.0))   # compute_distance_score
        vel = min(self.vel.get(d["id"], (0, 0, 0.0))[2] / 500.0, 1.0)
        comps = (s["ALPHA"] * base, s["BETA"] * conf, s["GAMMA"] * dist, s["DELTA"] * vel)
        return sum(comps), comps

    def select(self, dets):
        """Returns (detection or None, score, components, held)."""
        hold = self.spec.get("HOLD_FRAMES", 0)
        if hold and self.last_best is not None and not any(d["id"] == self.current_id for d in dets):
            self.missed += 1
            if self.missed <= hold:
                return self.last_best, self.current_score, (0, 0, 0, 0), True
        else:
            self.missed = 0
        best = self._select(dets)
        self.last_best = best[0]
        return (*best, False)

    def _select(self, dets):
        kind = self.spec["kind"]
        if not dets:
            return None, 0.0, (0, 0, 0, 0)
        if kind == "asps":
            self._update_velocity(dets)
            scored = [(d, *self._asps_score(d)) for d in dets]
            scored.sort(key=lambda x: x[1], reverse=True)
            best = scored[0]
            if self.current_id is not None and best[0]["id"] != self.current_id:
                cur = next((x for x in scored if x[0]["id"] == self.current_id), None)
                if cur is not None and best[1] < cur[1] + self.spec["SWITCH_THRESHOLD"]:
                    best = cur
            self.current_id, self.current_score = best[0]["id"], best[1]
            return best
        if kind == "class_priority":
            d = max(dets, key=lambda d: (CLASS_PRIORITIES.get(d["cls"], 0), d["conf"]))
        elif kind == "nearest":
            d = min(dets, key=self.proxy_distance)
        elif kind == "max_confidence":
            d = max(dets, key=lambda d: d["conf"])
        elif kind == "center_closest":
            d = min(dets, key=lambda d: math.hypot((d["xyxy"][0] + d["xyxy"][2]) / 2 - self.cx,
                                                   (d["xyxy"][1] + d["xyxy"][3]) / 2 - self.cy))
        else:
            raise ValueError(kind)
        self.current_id = d["id"]
        return d, 0.0, (0, 0, 0, 0)


class KF1D:
    """Constant-velocity KF identical to KalmanFilter in ASPS_LIDAR_TR.py."""

    def __init__(self, dt, Q, R, x0):
        self.x = np.array([[x0], [0.0]])
        self.P = np.diag([1000.0, 1000.0])
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[R]])
        self.A = np.array([[1.0, dt], [0.0, 1.0]])
        self.Q = np.array([[0.25 * dt ** 4, 0.5 * dt ** 3], [0.5 * dt ** 3, dt ** 2]]) * Q

    def step(self, z):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        if z is not None:
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0, 0])


# --- Ground truth + metrics --------------------------------------------------
def load_gt(path):
    gt = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            f = int(row["frame"])
            if row.get("x1", "").strip() == "":
                gt[f] = None
            else:
                gt[f] = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
    return gt


def replay_policy(name, spec, cache, fps, args, out_dir, gt):
    pol = Policy(name, spec, cache["width"], cache["height"], fps, args.focal_px, args.obj_h_m)
    kfx = KF1D(1.0 / fps, Q_KF, R_KF, cache["width"] / 2)
    kfy = KF1D(1.0 / fps, Q_KF, R_KF, cache["height"] / 2)
    rows, sel_ms = [], []
    prev_id = None
    for fr in cache["frames"]:
        t0 = time.perf_counter()
        best, score, comps, held = pol.select(fr["dets"])
        sel_ms.append((time.perf_counter() - t0) * 1000.0)
        if best is not None:
            x1, y1, x2, y2 = best["xyxy"]
            rx, ry = (x1 + x2) / 2, (y1 + y2) / 2
            if best["id"] != prev_id and not args.no_kf_reset:
                # new target: re-initialise the filters so they do not lag from the old one
                kfx = KF1D(1.0 / fps, Q_KF, R_KF, rx)
                kfy = KF1D(1.0 / fps, Q_KF, R_KF, ry)
            prev_id = best["id"]
            if held:   # coasting through a dropout: predict only
                kx, ky = kfx.step(None), kfy.step(None)
                rx = ry = ""
            else:
                kx, ky = kfx.step(np.array([[rx]])), kfy.step(np.array([[ry]]))
            rows.append([fr["frame"], best["id"], best["cls"], round(best["conf"], 4),
                         *[round(v, 2) for v in best["xyxy"]], round(score, 4),
                         *[round(c, 4) for c in comps],
                         rx if rx == "" else round(rx, 2), ry if ry == "" else round(ry, 2),
                         round(kx, 2), round(ky, 2), len(fr["dets"]), int(held)])
        else:
            kx, ky = kfx.step(None), kfy.step(None)
            rows.append([fr["frame"], -1, -1, 0, "", "", "", "", 0, 0, 0, 0, 0,
                         "", "", round(kx, 2), round(ky, 2), len(fr["dets"]), 0])
    header = ["frame", "track_id", "cls", "conf", "x1", "y1", "x2", "y2", "score",
              "baseP", "confP", "distP", "velP", "raw_cx", "raw_cy", "kf_cx", "kf_cy", "n_dets", "held"]
    with open(os.path.join(out_dir, f"sel_{name}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return compute_metrics(name, rows, cache, fps, gt, sel_ms)


def compute_metrics(name, rows, cache, fps, gt, sel_ms):
    n = len(rows)
    minutes = n / fps / 60.0 if n else float("nan")
    ids = [r[1] for r in rows]
    with_dets = [r for r in rows if r[17] > 0]

    # switches: selected track id changes between consecutive frames that both have a selection
    switches = sum(1 for a, b in zip(ids, ids[1:]) if a != -1 and b != -1 and a != b)
    # "flicker" switch: switching away and back to the same track within 1 s
    win = max(int(fps), 1)
    flicker = 0
    for i in range(1, n):
        if ids[i] != -1 and ids[i - 1] != -1 and ids[i] != ids[i - 1]:
            if ids[i - 1] in ids[i + 1:i + 1 + win]:
                flicker += 1

    # acquisition delay: a track whose class priority exceeds the currently-selected
    # target's appears -> frames until it is selected (censored at end of clip)
    first_seen, delays, missed = {}, [], 0
    frame_dets = {fr["frame"]: fr["dets"] for fr in cache["frames"]}
    for i, r in enumerate(rows):
        for d in frame_dets[r[0]]:
            if d["id"] in first_seen:
                continue
            first_seen[d["id"]] = i
            if i == 0 or rows[i - 1][1] == -1:
                continue
            cur_p = CLASS_PRIORITIES.get(rows[i - 1][2], 0)
            if CLASS_PRIORITIES.get(d["cls"], 0) > cur_p:
                hit = next((j for j in range(i, n) if ids[j] == d["id"]), None)
                if hit is None:
                    missed += 1
                else:
                    delays.append((hit - i) / fps)

    # Kalman jitter: std of frame-to-frame centre change, within constant-track segments
    raw_d, kf_d = [], []
    for a, b in zip(rows, rows[1:]):
        if a[1] != -1 and a[1] == b[1] and a[13] != "" and b[13] != "":
            raw_d.append(b[13] - a[13])
            kf_d.append(b[15] - a[15])
    raw_j = float(np.std(raw_d)) if raw_d else float("nan")
    kf_j = float(np.std(kf_d)) if kf_d else float("nan")

    m = {
        "policy": name, "frames": n, "duration_s": round(n / fps, 2),
        "coverage": round(sum(1 for r in with_dets if r[1] != -1) / len(with_dets), 4) if with_dets else float("nan"),
        "switches": switches, "switches_per_min": round(switches / minutes, 3) if minutes else float("nan"),
        "flicker_switches": flicker,
        "acq_events": len(delays) + missed, "acq_missed": missed,
        "acq_delay_mean_s": round(float(np.mean(delays)), 3) if delays else float("nan"),
        "raw_jitter_px": round(raw_j, 3), "kf_jitter_px": round(kf_j, 3),
        "kf_jitter_reduction_pct": round(100 * (1 - kf_j / raw_j), 2) if raw_d and raw_j > 0 else float("nan"),
        "select_ms_mean": round(float(np.mean(sel_ms)), 4) if sel_ms else float("nan"),
    }
    if gt:
        correct = total = 0
        for r in rows:
            if r[0] not in gt:
                continue
            total += 1
            g = gt[r[0]]
            if g is None:
                correct += int(r[1] == -1)
            elif r[1] != -1 and iou(r[4:8], g) >= 0.5:
                correct += 1
        m["gt_frames"] = total
        m["selection_accuracy"] = round(correct / total, 4) if total else float("nan")
        m["wrong_target_rate"] = round(1 - correct / total, 4) if total else float("nan")
    return m


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="video file or folder of frames")
    ap.add_argument("--cache", help="re-use a tracks.json from a previous run (skips detection)")
    ap.add_argument("--model", default=os.path.join(REPO_ROOT, "models", "best.pt"))
    ap.add_argument("--gt", help="ground-truth CSV: frame,x1,y1,x2,y2")
    ap.add_argument("--policies", default=",".join(POLICIES), help="comma-separated subset")
    ap.add_argument("--fps", type=float, default=30.0, help="fps for frame folders / fallback")
    ap.add_argument("--conf", type=float, default=_cfg.get("CONF_THRESHOLD", 0.5))
    ap.add_argument("--iou", type=float, default=_cfg.get("IOU_THRESHOLD", 0.4))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 inference (GPU/TensorRT)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--track-iou", type=float, default=0.3)
    ap.add_argument("--track-max-age", type=int, default=15)
    ap.add_argument("--focal-px", type=float, default=600.0, help="distance proxy focal length")
    ap.add_argument("--obj-h-m", type=float, default=2.0, help="distance proxy object height")
    ap.add_argument("--no-kf-reset", action="store_true",
                    help="do NOT re-init the KF on target switch (mimics the flight script)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    if not args.source and not args.cache:
        ap.error("give --source or --cache")

    if args.cache:
        with open(args.cache) as fh:
            cache = json.load(fh)
        fps = cache.get("fps", args.fps)
        base = args.name or os.path.basename(os.path.dirname(os.path.abspath(args.cache)))
    else:
        fps = source_fps(args.source, args.fps)
        print(f"Detecting on {args.source} with {args.model} ...")
        cache = run_detection(args)
        cache["fps"] = fps
        base = args.name or os.path.splitext(os.path.basename(os.path.normpath(args.source)))[0]

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, f"{base}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tracks.json"), "w") as fh:
        json.dump(cache, fh)

    gt = load_gt(args.gt) if args.gt else None
    summary = []
    for name in args.policies.split(","):
        summary.append(replay_policy(name, POLICIES[name], cache, fps, args, out_dir, gt))

    keys = list(dict.fromkeys(k for m in summary for k in m))
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)

    inf = [fr["infer_ms"] for fr in cache["frames"] if "infer_ms" in fr]
    meta = {
        "git_rev": git_rev(), "timestamp": stamp, "source": args.source, "cache": args.cache,
        "model": cache.get("model"), "class_names": cache.get("names"), "frames": len(cache["frames"]),
        "fps": fps, "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz, "half": args.half,
        "class_priorities": CLASS_PRIORITIES, "distance_proxy": {"focal_px": args.focal_px, "obj_h_m": args.obj_h_m},
        "tracker": {"type": "iou-hungarian", "iou": args.track_iou, "max_age": args.track_max_age},
        "infer_ms_mean": round(float(np.mean(inf)), 2) if inf else None,
        "infer_ms_p95": round(float(np.percentile(inf, 95)), 2) if inf else None,
        "infer_fps": round(1000.0 / float(np.mean(inf)), 2) if inf else None,
        "gt": args.gt,
    }
    import yaml
    with open(os.path.join(out_dir, "run_metadata.yaml"), "w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)

    cols = ["policy", "coverage", "switches_per_min", "flicker_switches", "acq_events",
            "acq_delay_mean_s", "kf_jitter_reduction_pct"] + (["selection_accuracy"] if gt else [])
    print(f"\nInference: {meta['infer_ms_mean']} ms mean, {meta['infer_ms_p95']} ms p95 "
          f"({meta['infer_fps']} FPS) over {meta['frames']} frames")
    print(" | ".join(cols))
    for m in summary:
        print(" | ".join(str(m.get(c, "")) for c in cols))
    print(f"\nResults: {out_dir}")


if __name__ == "__main__":
    main()
