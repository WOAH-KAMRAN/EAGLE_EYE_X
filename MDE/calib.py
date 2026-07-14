import cv2
from ultralytics import YOLO
import numpy as np

# ================= PARAMETERS =================
H_REAL = 15.0       # object height in cm
Z_REF = 30.0        # known distance in cm
PIXELS_SAMPLES = [] # store multiple measurements

# ================= LOAD YOLO =================
model = YOLO("yolov8n.pt")  # nano model for CPU

# ================= CAMERA =================
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def mouse_callback(event, x, y, flags, param):
    global PIXELS_SAMPLES, frame, results
    if event == cv2.EVENT_LBUTTONDOWN:
        if results.boxes:
            # pick largest box
            max_h = 0
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                h = y2 - y1
                if h > max_h:
                    max_h = h
            PIXELS_SAMPLES.append(max_h)
            print(f"Sample recorded: {max_h} px")

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", mouse_callback)

print("Calibration: click on the object in the image to record height in pixels.")
print("Take multiple clicks for averaging. Press ESC to finish.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detection
    results = model(frame, conf=0.3)[0]
    if results.boxes:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Calibration", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# ================= Compute F_PIXEL =================
if PIXELS_SAMPLES:
    h_avg = np.mean(PIXELS_SAMPLES)
    F_PIXEL = h_avg * Z_REF / H_REAL
    print(f"\nCalibration complete!")
    print(f"Average object height: {h_avg:.2f} px")
    print(f"Computed focal length: F_PIXEL = {F_PIXEL:.2f} px")
else:
    print("No samples recorded.")
