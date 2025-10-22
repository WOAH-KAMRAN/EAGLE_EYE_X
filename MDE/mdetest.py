import cv2
import torch
import numpy as np

# Load lightweight MiDaS model
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ---- Calibration ----
# Suppose depth value ~0.5 corresponds to 50 cm
# Adjust after testing with your setup
REFERENCE_DEPTH_VALUE = 0.5
REFERENCE_DISTANCE_CM = 50
scale = REFERENCE_DISTANCE_CM / REFERENCE_DEPTH_VALUE

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Get depth value at center
    h, w = depth_map.shape
    center_depth_value = depth_map[h//2, w//2]

    # Convert to real-world distance (cm)
    estimated_distance = center_depth_value * scale

    # Normalize depth map for display
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_vis = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype("uint8")
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    # Show estimated distance on webcam feed
    cv2.putText(frame, f"Distance: {estimated_distance:.1f} cm",
                (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)
    cv2.imshow("Depth Map", depth_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
