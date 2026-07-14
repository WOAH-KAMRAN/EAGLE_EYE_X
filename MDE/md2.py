import cv2
from ultralytics import YOLO

# ================= PARAMETERS =================
H_REAL = 20.0  # real object height in cm
F_PIXEL = 500  # focal length (compute once from reference)

# ================= LOAD YOLO MODEL =================
# Use yolov8n.pt for CPU-friendly detection
model = YOLO("yolov8n.pt")  

# ================= CAMERA =================
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ================= DETECTION =================
    results = model(frame, conf=0.3)[0]  # get first frame result
    # results.boxes.xyxy gives bounding boxes
    if results.boxes:
        for box in results.boxes.xyxy:  # loop over detected boxes
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # object height in pixels
            h_pixels = y2 - y1
            if h_pixels > 0:
                distance_cm = F_PIXEL * H_REAL / h_pixels
                cv2.putText(frame,
                            f"Dist: {distance_cm:.1f} cm",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2)

    cv2.imshow("Distance Estimation", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
