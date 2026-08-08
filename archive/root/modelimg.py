import cv2
import glob
import os
from ultralytics import YOLO

# --- SETTINGS ---
MODEL_PATH = "best.pt"                # Path to your trained YOLO model
INPUT_PATH = "images"                 # Folder containing input images OR a single image path
OUTPUT_PATH = "results"               # Folder to save annotated results
CONF_THRESHOLD = 0.5                  # Detection confidence threshold

# --- Load model ---
model = YOLO(MODEL_PATH)

# --- Prepare image list ---
if os.path.isdir(INPUT_PATH):
    image_paths = glob.glob(os.path.join(INPUT_PATH, "*.jpg")) + glob.glob(os.path.join(INPUT_PATH, "*.png"))
else:
    image_paths = [INPUT_PATH]

if not image_paths:
    print(f"❌ No images found in {INPUT_PATH}")
    exit()

os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"✅ Loaded {len(image_paths)} image(s). Running inference...")

# --- Run detection on each image ---
for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read {img_path}")
        continue

    # Run inference
    results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)
    annotated = results[0].plot()  # Get annotated image

    # Save output
    filename = os.path.basename(img_path)
    save_path = os.path.join(OUTPUT_PATH, filename)
    cv2.imwrite(save_path, annotated)
    print(f"🖼️ Saved result: {save_path}")

print(f"✅ All done! Annotated images saved in '{OUTPUT_PATH}/'")
