import cv2
from ultralytics import YOLO

# ================= PARAMETERS (CALIBRATED) =================
H_REAL = 20.0        # real object height in cm
F_PIXEL = 955.78     # calibrated focal length in pixels

# ================= LOAD YOLO =================
# yolov8n.pt = nano model, CPU-friendly
model = YOLO("yolov8n.pt")

# ================= CAMERA =================
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press ESC to quit. Tracking largest detected object...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ================= DETECTION =================
    results = model(frame, conf=0.3)[0]

    if results.boxes:
        # pick largest bounding box (assuming one main object)
        max_h = 0
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            h = y2 - y1
            if h > max_h:
                max_h = h
                best_box = (x1, y1, x2, y2)

        x1, y1, x2, y2 = best_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ================= COMPUTE DISTANCE =================
        h_pixels = y2 - y1
        if h_pixels > 0:
            distance_cm = F_PIXEL * H_REAL / h_pixels
            cv2.putText(frame,
                        f"Distance: {distance_cm:.1f} cm",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

    # ================= DISPLAY =================
    cv2.imshow("Distance Estimation", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
