import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA context

# --------------------------
# User Config
# --------------------------
ENGINE_PATH = "yolov8n.engine"  # <-- Update this path
USE_CSI = False  # True = RPi/CSI cam, False = USB webcam

CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 640, 480, 30

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

# --------------------------
# TensorRT Engine Loader
# --------------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

if engine is None:
    raise RuntimeError(f"❌ Failed to load engine {ENGINE_PATH}")

context = engine.create_execution_context()

# Allocate buffers
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

for binding in engine:
    binding_idx = engine.get_binding_index(binding)
    shape = engine.get_binding_shape(binding)
    size = trt.volume(shape)
    dtype = trt.nptype(engine.get_binding_dtype(binding))

    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)

    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
    else:
        outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

print("✅ Engine loaded. Input:", inputs[0]["shape"], "Outputs:", [o["shape"] for o in outputs])

# --------------------------
# Inference Loop
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Resize to model input size
    input_h, input_w = inputs[0]["shape"][2], inputs[0]["shape"][3]
    img = cv2.resize(frame, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)  # NCHW

    # Copy to device
    np.copyto(inputs[0]["host"], img.ravel())
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)

    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Copy outputs back
    results = []
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
        results.append(out["host"].reshape(out["shape"]))
    stream.synchronize()

    # For demo: just print first few values
    print("Output sample:", results[0].flatten()[:10])

    # Show camera
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
