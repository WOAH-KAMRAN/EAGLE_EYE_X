import cv2
from ultralytics import YOLO

# ================= LOAD MODEL =================
model = YOLO("yolov8n.pt")   # replace with your trained model if needed

# ================= VEHICLE CLASSES =================
VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bike", "bicycle"
}

# ================= CAMERA =================
cap = cv2.VideoCapture(2)

# Increased frame size (balanced for CPU)
FRAME_W = 640
FRAME_H = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

print("GREEN → FRIENDLY | RED → TRACKING")
print("ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ================= DETECTION =================
    results = model(frame, conf=0.35, verbose=False)[0]

    if results.boxes:
        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls_id)].lower()

            # ================= CLASS LOGIC =================
            if class_name in VEHICLE_CLASSES:
                color = (0, 0, 255)   # RED
                label = "TRACKING"
            else:
                color = (0, 255, 0)   # GREEN
                label = "FRIENDLY"

            # ================= DRAW BOX =================
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ================= LABEL BELOW BOX =================
            text_y = y2 + 22 if y2 + 22 < FRAME_H else y2 - 8
            cv2.putText(
                frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA
            )

    cv2.imshow("UAV Object Status", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
