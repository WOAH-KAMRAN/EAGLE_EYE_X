import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog

def click_event(event, x, y, flags, param):
    global calibration_points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(depth_vis, (x, y), 5, (255, 255, 255), -1)
        cv2.imshow("Depth Calibration", depth_vis)

        # Use a Tkinter popup dialog to safely get user input
        root = tk.Tk()
        root.withdraw()  # hide the main window
        try:
            value = simpledialog.askfloat(
                "Calibration Input",
                f"Measured distance at ({x}, {y}) [m]:",
                minvalue=0.0,
                maxvalue=50.0
            )
        finally:
            root.destroy()

        if value is not None:
            calibration_points.append((x, y, value))
            print(f"✔️  Added point ({x}, {y}) → {value} m")
        else:
            print("⚠️ Skipped point (no value entered)")


# --- Load MiDaS model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "MiDaS_small"  # options: "MiDaS_small", "DPT_Hybrid", "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device).eval()

# --- Load transforms ---
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if "DPT" in model_type:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# --- Helper functions ---
def estimate_depth(frame):
    """Runs MiDaS depth inference and returns raw inverse depth map (numpy)."""
    img_input = transform(frame).to(device)
    with torch.no_grad():
        prediction = midas(img_input)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map

def normalize_depth(depth_map):
    """Normalize for display."""
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    depth_vis = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
    return depth_color

# --- Calibration helper ---
def calibrate_depth(depth_map, calibration_points):
    """
    depth_map: raw MiDaS inverse depth map
    calibration_points: list of tuples (x, y, known_distance_m)
    Returns scale and offset for metric conversion: D = a / inv + b
    """
    inv_values = []
    true_distances = []
    for (x, y, d_true) in calibration_points:
        inv_values.append(depth_map[y, x])
        true_distances.append(d_true)
    inv_values = np.array(inv_values)
    true_distances = np.array(true_distances)

    X = np.vstack([1 / inv_values, np.ones_like(inv_values)]).T
    params, *_ = np.linalg.lstsq(X, true_distances, rcond=None)
    a, b = params
    print(f"✅ Calibration complete: a = {a:.3f}, b = {b:.3f}")
    return a, b

def convert_to_meters(depth_map, a, b):
    """Apply linear calibration to convert MiDaS inverse depth to meters."""
    distance_map = a / depth_map + b
    distance_map = np.clip(distance_map, 0, 20)  # limit to 20m for visualization
    return distance_map

# --- Main demo ---
if __name__ == "__main__":
    img_path = input("Enter path to an image (or leave blank for webcam): ").strip()

    if img_path == "":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam.")
            exit()
        ret, frame = cap.read()
        cap.release()
    else:
        frame = cv2.imread(img_path)
        if frame is None:
            print("❌ Could not load image.")
            exit()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_map = estimate_depth(frame_rgb)
    depth_vis = normalize_depth(depth_map)

    # Show depth map for calibration clicks
    print("\n🧭 Click on known-distance points in the depth map window.")
    print("After each click, enter the real distance in meters (e.g., 2.0).")
    print("Press 'q' when done.\n")

    calibration_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(depth_vis, (x, y), 5, (255, 255, 255), -1)
            cv2.imshow("Depth Calibration", depth_vis)
            d_true = float(input(f"Measured distance at ({x},{y}) [m]: "))
            calibration_points.append((x, y, d_true))

    cv2.imshow("Depth Calibration", depth_vis)
    cv2.setMouseCallback("Depth Calibration", click_event)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(calibration_points) < 2:
        print("❌ Need at least 2 calibration points for metric scaling.")
        exit()

    a, b = calibrate_depth(depth_map, calibration_points)
    metric_map = convert_to_meters(depth_map, a, b)
    metric_vis = normalize_depth(1 / metric_map)  # invert for color contrast

    # Show results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Calibrated Depth (m)")
    plt.imshow(metric_vis)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Save calibrated depth map
    cv2.imwrite("calibrated_depth.png", metric_vis)
    np.save("metric_depth.npy", metric_map)
    print("✅ Saved calibrated depth map: calibrated_depth.png")
    print("✅ Saved numeric depth data: metric_depth.npy")
