import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")  # <- Make sure best.pt is in the same folder or give full path

# Initialize webcam
cap = cv2.VideoCapture(2)  # Change index to 1 or 2 if another camera
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("✅ Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Run YOLO inference
    results = model.predict(frame, conf=0.5, verbose=False)

    # Annotate the frame with boxes and labels
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Detection stopped.")
