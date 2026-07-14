import cv2
from ultralytics import YOLO

# ================== CONFIG ==================
INFER_WIDTH = 416     # inference resolution (KEY)
INFER_HEIGHT = 416
FRAME_SKIP = 5        # detect once every N frames

VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorcycle", "bike", "bicycle"
}

# ================== INIT ==================
cv2.setUseOptimized(True)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
cached_boxes = []

print("CPU-safe mode ON | ESC to exit")

# ================== LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ---------- RUN YOLO SPARSELY ----------
    if frame_count % FRAME_SKIP == 0:
        resized = cv2.resize(frame, (INFER_WIDTH, INFER_HEIGHT))

        results = model(
            resized,
            conf=0.35,
            iou=0.5,
            device="cpu",
            verbose=False
        )[0]

        cached_boxes = []

        if results.boxes:
            for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = box.tolist()

                # scale back to original frame
                sx = frame.shape[1] / INFER_WIDTH
                sy = frame.shape[0] / INFER_HEIGHT

                x1 = int(x1 * sx)
                x2 = int(x2 * sx)
                y1 = int(y1 * sy)
                y2 = int(y2 * sy)

                class_name = model.names[int(cls_id)].lower()

                if class_name in VEHICLE_CLASSES:
                    cached_boxes.append((x1, y1, x2, y2, "TRACKING", (0, 0, 255)))
                else:
                    cached_boxes.append((x1, y1, x2, y2, "FRIENDLY", (0, 255, 0)))

    # ---------- DRAW CACHED RESULTS ----------
    for x1, y1, x2, y2, label, color in cached_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        ty = y2 + 20 if y2 + 20 < frame.shape[0] else y2 - 8
        cv2.putText(
            frame,
            label,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("CPU-SAFE UAV VISION", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
