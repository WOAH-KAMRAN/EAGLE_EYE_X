import cv2
import numpy as np
import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# -------------------------
# User Config
# -------------------------
CKPT_PATH = "/home/flanker/models/ZoeD_N.pt"  # <-- Download manually from HuggingFace
USE_CSI = False   # True = Jetson CSI cam, False = USB webcam

CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Camera Setup
# -------------------------
if USE_CSI:
    pipeline = (
        f"nvarguscamsrc sensor_id=0 ! "
        f"video/x-raw(memory:NVMM), width=(int){CAM_WIDTH}, height=(int){CAM_HEIGHT}, "
        f"format=(string)NV12, framerate=(fraction){CAM_FPS}/1 ! "
        "nvvidconv flip-method=0 ! "
        f"video/x-raw, width=(int){CAM_WIDTH}, height=(int){CAM_HEIGHT}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(0)  # USB webcam

if not cap.isOpened():
    raise RuntimeError("❌ Could not open camera")

# -------------------------
# Load ZoeDepth from local ckpt
# -------------------------
print(f"Loading ZoeDepth model from {CKPT_PATH} ...")
conf = get_config("zoedepth", "ZoeD_N")  # Nano variant
model = build_model(conf)
checkpoint = torch.load(CKPT_PATH, map_location="cpu")

# Some checkpoints store weights under 'state_dict'
state_dict = checkpoint.get("state_dict", checkpoint)
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()
print("✅ ZoeDepth loaded successfully.")

# -------------------------
# Depth Estimation Function
# -------------------------
def estimate_depth(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        depth = model.infer(img_tensor)[0]  # HxW tensor

    depth_norm = cv2.normalize(depth.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)
    return depth_norm.astype(np.uint8)

# -------------------------
# Main Loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    depth_map = estimate_depth(frame)
    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

    stacked = np.hstack((frame, depth_color))
    cv2.imshow("ZoeDepth: RGB | Depth", stacked)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
