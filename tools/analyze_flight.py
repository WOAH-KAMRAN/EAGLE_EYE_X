#!/usr/bin/env python3
"""Flight-log analysis: tracking-error metrics and paper-ready figures.

Usage:
    python tools/analyze_flight.py flights/flight_data_20250916_103253/flight_log_20250916_103253.csv
    python tools/analyze_flight.py flights/ -r                 # analyze all logs recursively
    python tools/analyze_flight.py log.csv --figdir paper/figs --summary-out summary.csv

Metrics (computed over OFFBOARD rows only):
  - yaw/pitch tracking error: RMSE, MAE, peak, P95
  - altitude error: RMSE, MAE
  - target switches: count of Class changes + time in lock
  - distance-following error (ASPS/lidar logs): RMSE, MAE of (KF_Dist_m - TARGET_DISTANCE_M)
  - ASPS component means (BaseP/ConfP/DistP/VelP/FinalP) when logged

Figures (one per log, into --figdir):
  - error_curves.png      yaw/pitch error + commands over time
  - target_switch.png     tracked Class / centered / mode over time
  - asps_components.png   ASPS component breakdown (requires ASPS columns)
  - distance_follow.png   LiDAR distance vs target (requires ASPS/lidar columns)
"""
import argparse
import glob
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Canonical column names mapped from the different CSV schemas that have been
# written over the project lifetime.
COLUMN_ALIASES = {
    "Time": ["Time", "Time_s", "t"],
    "Mode": ["Mode", "FlightMode"],
    "Alt": ["Alt", "Altitude"],
    "Yaw_Err": ["Yaw_Err", "Raw_Yaw_Err_px", "yaw_err"],
    "Pitch_Err": ["Pitch_Err", "Raw_Pitch_Align_Err_px", "pitch_err"],
    "Yaw_Cmd": ["Yaw_Cmd", "Yaw_Cmd_deg_s", "yaw_cmd"],
    "Pitch_Cmd": ["Pitch_Cmd", "Pitch_Cmd_deg_s", "pitch_cmd"],
    "Class": ["Class", "Class_ID", "Target_Class_ID", "cls"],
    "Conf": ["Conf", "Confidence"],
    "Target_Centered": ["Target_Centered", "centered"],
    "KF_Dist_m": ["KF_Dist_m", "kf_dist_m", "Dist_m"],
    "Alt_Err": ["Alt_Err", "alt_err"],
    "KF_dt": ["KF_dt", "dt"],
    "Raw_LiDAR_m": ["Raw_LiDAR_m", "raw_lidar_m"],
    "BaseP": ["BaseP"], "ConfP": ["ConfP"],
    "DistP": ["DistP"], "VelP": ["VelP"], "FinalP": ["FinalP"],
}

TARGET_DISTANCE_DEFAULT = 2.0  # m, used when Alt/Target columns absent


def _norm(df):
    renamed = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                renamed[a] = canonical
                break
    return df.rename(columns=renamed)


def _rmse(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(math.sqrt((s ** 2).mean()))


def _mae(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(s.abs().mean())


def _p95(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(s.abs().quantile(0.95))


def analyze_log(path, target_distance=TARGET_DISTANCE_DEFAULT):
    df = pd.read_csv(path)
    df = _norm(df)
    t = pd.to_numeric(df.get("Time"), errors="coerce")
    t0 = t.dropna().min() if "Time" in df else 0.0
    df["t_rel"] = (t - t0).fillna(0.0)

    mode = df.get("Mode").astype(str).str.upper()
    offboard = mode == "OFFBOARD"

    metric = {}
    metric["file"] = os.path.basename(path)
    metric["rows"] = len(df)
    metric["offboard_rows"] = int(offboard.sum())
    metric["duration_s"] = float(df["t_rel"].max()) if "Time" in df else float("nan")

    if "Time" in df:
        metric["fps"] = metric["rows"] / metric["duration_s"] if metric["duration_s"] else float("nan")

    # --- Tracking error (OFFBOARD) ---
    for col, key in [("Yaw_Err", "yaw"), ("Pitch_Err", "pitch")]:
        if col in df:
            s = df.loc[offboard, col]
            metric[f"{key}_rmse_deg_px"] = _rmse(s)
            metric[f"{key}_mae_deg_px"] = _mae(s)
            metric[f"{key}_p95"] = _p95(s)

    # Altitude error
    if "Alt_Err" in df:
        metric["alt_rmse_m"] = _rmse(df.loc[offboard, "Alt_Err"])
        metric["alt_mae_m"] = _mae(df.loc[offboard, "Alt_Err"])
    elif "Alt" in df:
        s = pd.to_numeric(df.loc[offboard, "Alt"], errors="coerce")
        metric["alt_rmse_m"] = _rmse(s - 2.0)
        metric["alt_mae_m"] = _mae(s - 2.0)

    # --- Target lock / switching ---
    if "Class" in df:
        cls = pd.to_numeric(df["Class"], errors="coerce")
        switches = int((cls.diff() != 0).sum())
        metric["target_switches"] = switches
        locked = (cls.notna()) & (cls >= 0)
        metric["lock_seconds"] = float(df.loc[locked & offboard, "t_rel"].max() -
                                       df.loc[locked & offboard, "t_rel"].min()) if locked.any() else 0.0
    if "Target_Centered" in df:
        centered = df["Target_Centered"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
        metric["centered_frac"] = float(centered.loc[offboard].mean()) if offboard.any() else float("nan")

    # --- Distance following ---
    if "KF_Dist_m" in df:
        s = pd.to_numeric(df.loc[offboard, "KF_Dist_m"], errors="coerce")
        metric["dist_rmse_m"] = _rmse(s - target_distance)
        metric["dist_mae_m"] = _mae(s - target_distance)

    # --- ASPS components ---
    if "FinalP" in df:
        for col in ["BaseP", "ConfP", "DistP", "VelP", "FinalP"]:
            metric[f"asps_{col}"] = float(pd.to_numeric(df[col], errors="coerce").mean())

    return df, offboard, metric


def plot_log(df, offboard, path, figdir, target_distance=TARGET_DISTANCE_DEFAULT):
    os.makedirs(figdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    t = df["t_rel"].where(offboard)

    # 1. error curves
    has_err = any(c in df for c in ("Yaw_Err", "Pitch_Err"))
    if has_err:
        fig, ax = plt.subplots(figsize=(10, 4))
        for col, label, color in [("Yaw_Err", "yaw error", "tab:red"),
                                  ("Pitch_Err", "pitch error", "tab:blue")]:
            if col in df:
                ax.plot(t, df[col], label=label, color=color, alpha=0.8)
        ax.set_xlabel("time (s)"); ax.set_ylabel("error (px)"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(f"{base} — tracking error")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{base}_error_curves.png"), dpi=150)
        plt.close(fig)

    # 2. target / mode switching
    if "Class" in df or "Mode" in df:
        fig, ax = plt.subplots(figsize=(10, 4))
        if "Class" in df:
            ax.plot(t, df["Class"], drawstyle="steps-post", label="tracked class", color="tab:green")
        mode_num = pd.factorize(df.get("Mode", pd.Series(index=df.index)))[0]
        ax.plot(df["t_rel"], mode_num, alpha=0.4, color="gray", label="mode (coded)")
        ax.set_xlabel("time (s)"); ax.set_ylabel("class / mode"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(f"{base} — target & mode over time")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{base}_target_switch.png"), dpi=150)
        plt.close(fig)

    # 3. ASPS components
    if "FinalP" in df:
        fig, ax = plt.subplots(figsize=(10, 5))
        for col, color in [("BaseP", "tab:blue"), ("ConfP", "tab:orange"),
                           ("DistP", "tab:green"), ("VelP", "tab:purple")]:
            if col in df:
                ax.plot(df["t_rel"], df[col], label=col, color=color, alpha=0.8)
        ax.plot(df["t_rel"], df["FinalP"], label="FinalP", color="black", linewidth=2)
        ax.set_xlabel("time (s)"); ax.set_ylabel("score"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(f"{base} — ASPS priority components")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{base}_asps_components.png"), dpi=150)
        plt.close(fig)

    # 4. distance following
    if "KF_Dist_m" in df:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, df["KF_Dist_m"], label="KF distance", color="tab:blue")
        ax.axhline(target_distance, color="tab:red", linestyle="--", label=f"target {target_distance:.1f} m")
        if "Raw_LiDAR_m" in df:
            ax.plot(t, df["Raw_LiDAR_m"], label="raw LiDAR", color="tab:green", alpha=0.5)
        ax.set_xlabel("time (s)"); ax.set_ylabel("distance (m)"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title(f"{base} — distance following")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, f"{base}_distance_follow.png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Flight-log metrics and paper figures")
    ap.add_argument("source", help="log CSV, or directory searched recursively with -r")
    ap.add_argument("-r", "--recursive", action="store_true", help="treat source as dir, search **/*.csv")
    ap.add_argument("--figdir", default="results/figures")
    ap.add_argument("--summary-out", default=None, help="write summary CSV")
    ap.add_argument("--target-distance", type=float, default=TARGET_DISTANCE_DEFAULT)
    args = ap.parse_args()

    paths = []
    if os.path.isdir(args.source):
        paths = sorted(glob.glob(os.path.join(args.source, "**", "*.csv"), recursive=args.recursive))
    else:
        paths = [args.source]
    paths = [p for p in paths if not p.endswith((".~lock",))]

    if not paths:
        raise SystemExit(f"No CSV logs found at {args.source}")

    rows = []
    for p in paths:
        try:
            df, offboard, metric = analyze_log(p, args.target_distance)
            plot_log(df, offboard, p, args.figdir, args.target_distance)
            rows.append(metric)
            print(f"analyzed {p}")
        except Exception as e:
            print(f"SKIP {p}: {e}", file=sys.stderr)

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n===== Summary =====")
    print(summary.to_string(index=False))

    if args.summary_out:
        summary.to_csv(args.summary_out, index=False)
        print(f"\nSummary written to {args.summary_out}")
    print(f"\nFigures saved to {args.figdir}/")


if __name__ == "__main__":
    main()
