# UAV Target Tracking System — ASPS LiDAR Following
# Adaptive Semantic Priority Selection (ASPS) + TF Mini LiDAR
#   - Per-frame KF dt | Display thread | DistanceKF + DistancePID
#   - ASPS: mission hierarchy + confidence + distance + velocity scoring
#   - Mock LiDAR mode for SITL testing (no hardware needed)
#
# For Jetson Orin Nano + TF Mini LiDAR (USB-UART)
# LiDAR port: /dev/ttyUSB0, 115200 baud (configurable below)
#
# Toggle ASPS: set USE_ASPS = False in prioritization.py for pure CLASS_PRIORITIES baseline

import time
import cv2
import numpy as np
import datetime
import os
import asyncio
import threading
import queue
from mavsdk import System
from mavsdk.offboard import OffboardError, AttitudeRate, VelocityNedYaw
from mavsdk.telemetry import FlightMode, RcStatus
import csv
import signal
import sys
import random

from prioritization import ASPSPrioritySelector, USE_ASPS

# PyTorch and YOLO imports
try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
    print(f"PyTorch imported. CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False
    sys.exit(1)

# Serial for LiDAR (optional — mock mode works without it)
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    print("pyserial not installed — LiDAR will use mock mode")
    SERIAL_AVAILABLE = False

# --- Configuration ---
CONNECTION_STRING = 'udpin://127.0.0.1:14550'

YOLO_MODEL_PATH = 'yolov8n.pt'

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

CLASS_PRIORITIES = {2: 4, 1: 3, 3: 2, 0: 1}

# Yaw/pitch centering
YAW_LIMIT = 30
PITCH_LIMIT = 15
ROLL_LIMIT = 15
HOVER_THRUST = 0.5
TAKEOFF_THRUST = 0.7

DEAD_ZONE_YAW = 20
DEAD_ZONE_PITCH = 20

# Altitude control
TARGET_ALTITUDE = 2.0
ALTITUDE_TOLERANCE = 0.3
MAX_ALTITUDE = 10.0
MIN_ALTITUDE = 0.5

# PID for centering (yaw + pitch)
KP_YAW = 0.03
KI_YAW = 0.001
KD_YAW = 0.01

KP_PITCH = 0.02
KI_PITCH = 0.001
KD_PITCH = 0.01

KP_ALT = 0.5
KI_ALT = 0.05
KD_ALT = 0.1

# Kalman Filter for centering
Q_KF = 0.1
R_KF = 10
KF_DT_DEFAULT = 0.033

# --- LiDAR Configuration ---
USE_REAL_LIDAR = False           # False = mock mode (no hardware needed)
LIDAR_PORT = '/dev/ttyUSB0'      # TF Mini serial port
LIDAR_BAUDRATE = 115200
LIDAR_MIN_RANGE_M = 0.1
LIDAR_MAX_RANGE_M = 12.0

# --- Distance Following Configuration ---
TARGET_DISTANCE_M = 2.0          # Desired follow distance
DISTANCE_DEAD_ZONE_M = 0.15      # ±15cm deadband
YAW_CENTERED_THRESHOLD = 30      # Pixels — target is "centered" when yaw err <= this
MAX_FORWARD_SPEED_MPS = 2.0      # Safety cap on forward velocity

# PID for distance (forward velocity control)
KP_DIST = 1.0
KI_DIST = 0.05
KD_DIST = 0.3

# Distance Kalman Filter
Q_DIST_KF = 0.1
R_DIST_KF = 3.0                  # LiDAR is accurate — low measurement noise

# Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_INDEX = 2

RECORD_VIDEO = True
VIDEO_FOLDER = "flights"

TELEMETRY_TIMEOUT = 2.0
DETECTION_TIMEOUT = 5.0
TAKEOFF_TIMEOUT = 20.0

EMERGENCY_STOP = False

display_queue = queue.Queue(maxsize=2)
display_should_stop = threading.Event()


# --- PID Controller ---
class PID:
    def __init__(self, Kp, Ki, Kd, output_limit=None, integrator_limit=100):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limit = output_limit
        self.integrator_limit = integrator_limit
        self.I_term = 0
        self.last_error = 0
        self.last_time = time.time()

    def update(self, error, dt=None):
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
        if dt <= 0:
            dt = 0.01
        P = self.Kp * error
        self.I_term += self.Ki * error * dt
        self.I_term = np.clip(self.I_term, -self.integrator_limit, self.integrator_limit)
        D = self.Kd * (error - self.last_error) / dt
        self.last_error = error
        output = P + self.I_term + D
        if self.output_limit:
            output = np.clip(output, -self.output_limit, self.output_limit)
        return output

    def reset(self):
        self.I_term = 0
        self.last_error = 0
        self.last_time = time.time()


# --- Kalman Filter for centering (yaw/pitch) ---
class KalmanFilter:
    def __init__(self, dt=KF_DT_DEFAULT, Q=0.1, R=10, initial_pos=0.0):
        self.dt = dt
        self.Q = Q
        self.R = R
        self.x = np.array([[initial_pos], [0.0]])
        self.P = np.array([[1000.0, 0.0], [0.0, 1000.0]])
        self.H = np.array([[1, 0]])
        self.R_matrix = np.array([[R]])
        self.A = np.array([[1, dt], [0, 1]])
        self.Q_matrix = np.array([[0.25 * dt**4, 0.5 * dt**3],
                                  [0.5 * dt**3, dt**2]]) * Q

    def update_dt(self, dt):
        if dt <= 0:
            dt = 0.01
        self.dt = dt
        self.A = np.array([[1, dt], [0, 1]])
        self.Q_matrix = np.array([[0.25 * dt**4, 0.5 * dt**3],
                                  [0.5 * dt**3, dt**2]]) * self.Q

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q_matrix

    def update(self, measurement):
        y = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R_matrix
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    def get_state(self):
        return float(self.x[0, 0]), float(self.x[1, 0])

    def reset(self, pos=0.0):
        self.x = np.array([[pos], [0.0]])
        self.P = np.array([[1000.0, 0.0], [0.0, 1000.0]])


# --- Distance Kalman Filter (1D distance + velocity) ---
class DistanceKalmanFilter:
    def __init__(self, dt=KF_DT_DEFAULT, Q=0.1, R=3.0, initial_distance=5.0):
        self.dt = dt
        self.Q = Q
        self.R = R
        self.x = np.array([[initial_distance], [0.0]])
        self.P = np.array([[100.0, 0.0], [0.0, 100.0]])
        self.H = np.array([[1, 0]])
        self.R_matrix = np.array([[R]])
        self.A = np.array([[1, dt], [0, 1]])
        self.Q_matrix = np.array([[0.25 * dt**4, 0.5 * dt**3],
                                  [0.5 * dt**3, dt**2]]) * Q

    def update_dt(self, dt):
        if dt <= 0:
            dt = 0.01
        self.dt = dt
        self.A = np.array([[1, dt], [0, 1]])
        self.Q_matrix = np.array([[0.25 * dt**4, 0.5 * dt**3],
                                  [0.5 * dt**3, dt**2]]) * self.Q

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q_matrix

    def update(self, measurement):
        if measurement is not None:
            y = measurement - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R_matrix
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(2) - K @ self.H) @ self.P

    def get_state(self):
        return float(self.x[0, 0]), float(self.x[1, 0])

    def reset(self, initial_distance=5.0):
        self.x = np.array([[initial_distance], [0.0]])
        self.P = np.array([[100.0, 0.0], [0.0, 100.0]])


# --- YOLO Detector ---
class YOLODetector:
    def __init__(self, model_path):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.device = 'cuda'
            print("  Using CUDA")
        else:
            self.device = 'cpu'
            print("  Using CPU (slower)")
        self.model.to(self.device)
        dummy = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        _ = self.model.predict(dummy, verbose=False, conf=0.5)
        print("  YOLO ready")

    def detect(self, frame, conf=0.5, iou=0.4):
        results = self.model.predict(frame, conf=conf, iou=iou, verbose=False)
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                detections.append({
                    'xyxy': [float(x1), float(y1), float(x2), float(y2)],
                    'cls': int(boxes.cls[i].cpu().numpy()),
                    'conf': float(boxes.conf[i].cpu().numpy())
                })
        return detections


# --- TF Mini LiDAR ---
class TFMiniLidar:
    def __init__(self, use_real=False, port=LIDAR_PORT, baud=LIDAR_BAUDRATE,
                 min_range=LIDAR_MIN_RANGE_M, max_range=LIDAR_MAX_RANGE_M):
        self.use_real = use_real and SERIAL_AVAILABLE
        self.port = port
        self.baud = baud
        self.min_range = min_range
        self.max_range = max_range
        self.serial_port = None
        self.last_valid_distance = None

    def initialize(self):
        if self.use_real:
            try:
                self.serial_port = serial.Serial(self.port, self.baud, timeout=1)
                print(f"  LiDAR serial port {self.port} opened successfully")
                return True
            except serial.SerialException as e:
                print(f"  Failed to open LiDAR port {self.port}: {e}")
                print("  Falling back to mock mode")
                self.use_real = False
                return False
        else:
            print("  LiDAR in MOCK mode — simulated distance data")
            return True

    def read_distance(self):
        if self.use_real and self.serial_port and self.serial_port.is_open:
            return self._read_real()
        else:
            return self._read_mock()

    def _read_real(self):
        try:
            self.serial_port.flushInput()
            while self.serial_port.in_waiting >= 9:
                if self.serial_port.read(1) == b'Y':
                    if self.serial_port.read(1) == b'Y':
                        frame_data = self.serial_port.read(7)
                        dist_cm = int.from_bytes(frame_data[0:2], byteorder='little')
                        strength = int.from_bytes(frame_data[2:4], byteorder='little')
                        dist_m = dist_cm / 100.0
                        if self.min_range <= dist_m <= self.max_range and strength > 10:
                            self.last_valid_distance = dist_m
                            return dist_m
            return None
        except Exception as e:
            print(f"  LiDAR read error: {e}")
            return None

    def _read_mock(self):
        if self.last_valid_distance is None:
            base = TARGET_DISTANCE_M + random.uniform(-1.0, 1.0)
        else:
            base = self.last_valid_distance + random.uniform(-0.05, 0.05)
        noise = random.uniform(-0.02, 0.02)
        distance = max(self.min_range, min(self.max_range, base + noise))
        self.last_valid_distance = distance
        return distance

    def cleanup(self):
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
                print("  LiDAR serial port closed")
            except Exception:
                pass


# --- Display Thread ---
def display_thread_func():
    print("[Display Thread] Starting...")
    cv2.namedWindow("ASPS LiDAR Tracking", cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty("ASPS LiDAR Tracking",
                              cv2.WND_PROP_OPENGL,
                              cv2.WINDOW_OPENGL)
    except Exception:
        pass
    while not display_should_stop.is_set():
        try:
            frame = display_queue.get(timeout=0.1)
            if frame is None:
                break
            cv2.imshow("ASPS LiDAR Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                display_should_stop.set()
                break
        except queue.Empty:
            cv2.waitKey(1)
            continue
        except Exception as e:
            print(f"[Display Thread] Error: {e}")
            break
    cv2.destroyAllWindows()
    print("[Display Thread] Stopped")


# --- Pre-flight Checks ---
async def preflight_checks(drone):
    print("\n  Running companion computer checks...")
    print("  Checking drone connection...")
    try:
        health = await asyncio.wait_for(
            drone.telemetry.health().__anext__(), timeout=5.0
        )
        print("    Telemetry connection OK")
    except asyncio.TimeoutError:
        print("    Cannot get telemetry")
        return False
    print("  Checking RC connection...")
    try:
        rc_status = await asyncio.wait_for(
            drone.telemetry.rc_status().__anext__(), timeout=5.0
        )
        if not rc_status.is_available:
            print("    RC not connected — needed for safety override")
            return False
        print(f"    RC OK (signal: {rc_status.signal_strength_percent}%)")
    except asyncio.TimeoutError:
        print("    RC check timeout")
        return False
    print("  Companion computer checks passed")
    print("  Ensure FC pre-arm checks pass before arming\n")
    return True


# --- Arm and Takeoff ---
async def arm_and_takeoff(drone, target_alt):
    print(f"\n  Starting takeoff to {target_alt}m...")
    print("  Setting HOLD mode...")
    await asyncio.sleep(1)
    try:
        await drone.action.hold()
        await asyncio.sleep(1)
    except Exception as e:
        print(f"    HOLD mode failed: {e}")
    print("  Arming motors...")
    try:
        await drone.action.arm()
        await asyncio.sleep(2)
        print("    Armed")
    except Exception as e:
        print(f"    Arming failed: {e}")
        return False
    await drone.offboard.set_attitude_rate(AttitudeRate(0, 0, 0, TAKEOFF_THRUST))
    try:
        await drone.offboard.start()
        print("    Offboard mode started")
    except OffboardError as e:
        print(f"    Offboard failed: {e}")
        return False
    print("  Taking off...")
    start_time = time.time()
    while time.time() - start_time < TAKEOFF_TIMEOUT:
        if EMERGENCY_STOP:
            return False
        try:
            pos = await asyncio.wait_for(
                drone.telemetry.position().__anext__(), timeout=2.0
            )
            alt = pos.relative_altitude_m
        except Exception:
            print("    Telemetry lost during takeoff")
            return False
        print(f"    Altitude: {alt:.2f}m / {target_alt:.2f}m")
        if alt >= target_alt * 0.95:
            print(f"    Reached {alt:.2f}m")
            break
        if alt >= MAX_ALTITUDE:
            print(f"    Max altitude reached")
            break
        await drone.offboard.set_attitude_rate(AttitudeRate(0, 0, 0, TAKEOFF_THRUST))
        await asyncio.sleep(0.2)
    else:
        print("    Takeoff timeout")
        return False
    await drone.offboard.set_attitude_rate(AttitudeRate(0, 0, 0, HOVER_THRUST))
    print("    Hovering\n")
    return True


# --- Main Tracking Loop ---
async def tracking_loop():
    global EMERGENCY_STOP

    print("=" * 60)
    print("UAV TARGET TRACKING — ASPS LIDAR FOLLOWING")
    asps_status = "ON" if USE_ASPS else "OFF (legacy baseline)"
    print(f"  ASPS: {asps_status} | YOLO + TF Mini LiDAR | Display thread")
    print("=" * 60)

    # Connect to drone
    drone = System()
    print(f"\nConnecting to drone: {CONNECTION_STRING}")
    await drone.connect(system_address=CONNECTION_STRING)
    print("Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected\n")
            break
        await asyncio.sleep(1)

    if not await preflight_checks(drone):
        print("Pre-flight checks failed. Aborting.")
        return

    print("Waiting for GUIDED/OFFBOARD mode...")
    print("  Switch to GUIDED mode in your ground station\n")
    while True:
        try:
            fm = await asyncio.wait_for(
                drone.telemetry.flight_mode().__anext__(), timeout=5.0
            )
            if fm == FlightMode.OFFBOARD:
                print("Ready for offboard control\n")
                break
            print(f"  Current mode: {fm.name}")
            await asyncio.sleep(2)
        except asyncio.TimeoutError:
            print("Mode check timeout")
            return

    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    if not cap.isOpened():
        print("Camera failed to open")
        return
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from camera")
        cap.release()
        return
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    print(f"Camera ready ({w}x{h})\n")

    # Initialize YOLO
    try:
        detector = YOLODetector(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"YOLO failed: {e}")
        cap.release()
        return

    # Initialize LiDAR
    lidar = TFMiniLidar(use_real=USE_REAL_LIDAR)
    lidar.initialize()

    # Initialize ASPS priority selector
    selector = ASPSPrioritySelector()

    # Start display thread
    display_thread = threading.Thread(target=display_thread_func, daemon=True)
    display_thread.start()
    print("Display thread started\n")

    # Takeoff
    if not await arm_and_takeoff(drone, TARGET_ALTITUDE):
        print("Takeoff failed")
        lidar.cleanup()
        cap.release()
        return

    # Initialize controllers
    yaw_pid = PID(KP_YAW, KI_YAW, KD_YAW, output_limit=YAW_LIMIT)
    pitch_pid = PID(KP_PITCH, KI_PITCH, KD_PITCH, output_limit=PITCH_LIMIT)
    alt_pid = PID(KP_ALT, KI_ALT, KD_ALT, output_limit=0.3)

    # Distance PID — maps distance error (m) → forward velocity (m/s)
    dist_pid = PID(KP_DIST, KI_DIST, KD_DIST, output_limit=MAX_FORWARD_SPEED_MPS)

    # Kalman filters
    yaw_kf = KalmanFilter(dt=KF_DT_DEFAULT, Q=Q_KF, R=R_KF, initial_pos=cx)
    pitch_kf = KalmanFilter(dt=KF_DT_DEFAULT, Q=Q_KF, R=R_KF, initial_pos=cy)
    dist_kf = DistanceKalmanFilter(dt=KF_DT_DEFAULT, Q=Q_DIST_KF, R=R_DIST_KF,
                                   initial_distance=TARGET_DISTANCE_M)

    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    flight_dir = os.path.join(VIDEO_FOLDER, f"asps_lidar_{timestamp}")
    os.makedirs(flight_dir, exist_ok=True)

    log_file = open(os.path.join(flight_dir, f"log_{timestamp}.csv"), 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        'Time', 'Mode', 'Alt', 'Alt_Err',
        'Yaw_Cmd', 'Pitch_Cmd', 'Raw_LiDAR_m', 'KF_Dist_m',
        'Dist_Err_m', 'Fwd_Vel_mps', 'Target_Centered',
        'Class', 'Conf', 'RC_OK', 'KF_dt',
        'BaseP', 'ConfP', 'DistP', 'VelP', 'FinalP'
    ])

    video_writer = None
    if RECORD_VIDEO:
        vid_path = os.path.join(flight_dir, f"video_{timestamp}.mp4")
        video_writer = cv2.VideoWriter(
            vid_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            CAMERA_FPS,
            (w, h)
        )

    print("Starting tracking...\n")
    print("Controls:")
    print("  Q or Ctrl+C: Emergency stop and land")
    print("  Switch to LOITER/HOLD in GCS to pause tracking\n")

    is_offboard = True
    last_detection_time = time.time()
    frame_count = 0
    last_frame_time = time.time()

    # ASPS component breakdown for logging
    comp_base = comp_conf = comp_dist = comp_vel = 0.0
    final_score = 0.0

    try:
        while not EMERGENCY_STOP:
            frame_count += 1

            # Per-frame dt
            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now

            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            # Get telemetry
            try:
                mode = await asyncio.wait_for(
                    drone.telemetry.flight_mode().__anext__(), timeout=TELEMETRY_TIMEOUT
                )
                pos = await asyncio.wait_for(
                    drone.telemetry.position().__anext__(), timeout=TELEMETRY_TIMEOUT
                )
                altitude = pos.relative_altitude_m
                rc = await asyncio.wait_for(
                    drone.telemetry.rc_status().__anext__(), timeout=TELEMETRY_TIMEOUT
                )
                rc_ok = rc.is_available
            except asyncio.TimeoutError:
                print("Telemetry timeout — landing for safety")
                break
            except Exception as e:
                print(f"Telemetry error: {e}")
                break

            if not rc_ok:
                print("RC connection lost — landing")
                break
            if altitude > MAX_ALTITUDE:
                print(f"Max altitude {MAX_ALTITUDE}m exceeded")
                break
            if altitude < MIN_ALTITUDE:
                print(f"Below minimum altitude {MIN_ALTITUDE}m")
                break

            yaw_cmd = pitch_cmd = 0.0
            forward_velocity = 0.0
            tracked_class = -1
            confidence = 0.0
            target_centered = False
            raw_lidar_m = 0.0
            kf_dist_m = 0.0
            dist_err_m = 0.0

            if mode == FlightMode.OFFBOARD:
                if not is_offboard:
                    try:
                        await drone.offboard.start()
                        is_offboard = True
                        yaw_pid.reset()
                        pitch_pid.reset()
                        alt_pid.reset()
                        dist_pid.reset()
                    except OffboardError:
                        pass

                # YOLO detection
                detections = detector.detect(frame, CONF_THRESHOLD, IOU_THRESHOLD)

                # Update Kalman filters with per-frame dt
                yaw_kf.update_dt(dt)
                pitch_kf.update_dt(dt)
                dist_kf.update_dt(dt)
                yaw_kf.predict()
                pitch_kf.predict()
                dist_kf.predict()

                # Select best target using ASPS
                def distance_fn(det):
                    return raw_lidar_m if target_centered else None

                best_det, components = selector.select(
                    detections, time.time(), CLASS_PRIORITIES, distance_fn
                )

                if components is not None:
                    comp_base = components['base_priority']
                    comp_conf = components['confidence']
                    comp_dist = components['distance']
                    comp_vel  = components['velocity']
                    final_score = comp_base + comp_conf + comp_dist + comp_vel
                else:
                    comp_base = comp_conf = comp_dist = comp_vel = 0.0
                    final_score = 0.0

                if best_det:
                    last_detection_time = time.time()

                    x1, y1, x2, y2 = best_det['xyxy']
                    raw_x = (x1 + x2) / 2
                    raw_y = (y1 + y2) / 2

                    yaw_kf.update(np.array([[raw_x]]))
                    pitch_kf.update(np.array([[raw_y]]))

                    est_x, _ = yaw_kf.get_state()
                    est_y, _ = pitch_kf.get_state()

                    yaw_err = est_x - cx
                    pitch_err = est_y - cy

                    # Step 1: Center the target (yaw + pitch)
                    if abs(yaw_err) > DEAD_ZONE_YAW:
                        yaw_cmd = yaw_pid.update(yaw_err)
                    else:
                        yaw_cmd = 0.0
                        yaw_pid.reset()

                    if abs(pitch_err) > DEAD_ZONE_PITCH:
                        pitch_cmd = pitch_pid.update(pitch_err)
                    else:
                        pitch_cmd = 0.0
                        pitch_pid.reset()

                    # Step 2: Check if target is centered enough for LiDAR
                    target_centered = abs(yaw_err) <= YAW_CENTERED_THRESHOLD

                    # Step 3: If centered, read LiDAR and move forward
                    if target_centered:
                        raw_lidar = lidar.read_distance()
                        if raw_lidar is not None:
                            raw_lidar_m = raw_lidar
                            dist_kf.update(np.array([[raw_lidar]]))

                        kf_dist, _ = dist_kf.get_state()
                        kf_dist_m = kf_dist

                        dist_err_m = kf_dist - TARGET_DISTANCE_M
                        if abs(dist_err_m) > DISTANCE_DEAD_ZONE_M:
                            forward_velocity = dist_pid.update(dist_err_m)
                            forward_velocity = np.clip(
                                forward_velocity,
                                -MAX_FORWARD_SPEED_MPS,
                                MAX_FORWARD_SPEED_MPS
                            )
                        else:
                            forward_velocity = 0.0
                            dist_pid.reset()

                    tracked_class = best_det['cls']
                    confidence = best_det['conf']

                    # Draw detections — ASPS selected target in cyan
                    for det in detections:
                        dx1, dy1, dx2, dy2 = [int(x) for x in det['xyxy']]
                        if det == best_det:
                            color = (255, 255, 0)  # Cyan for ASPS-selected
                        else:
                            color = (100, 100, 255)  # Red for others
                        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
                        label = f"{det['cls']}:{det['conf']:.2f}"
                        cv2.putText(frame, label, (dx1, dy1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    cv2.circle(frame, (int(raw_x), int(raw_y)), 5, (0, 0, 255), -1)
                    cv2.circle(frame, (int(est_x), int(est_y)), 8, (255, 0, 255), 2)

                else:
                    if time.time() - last_detection_time > DETECTION_TIMEOUT:
                        yaw_cmd = pitch_cmd = 0.0
                        forward_velocity = 0.0
                        yaw_pid.reset()
                        pitch_pid.reset()
                        dist_pid.reset()
                    else:
                        est_x, _ = yaw_kf.get_state()
                        est_y, _ = pitch_kf.get_state()
                        yaw_err = est_x - cx
                        pitch_err = est_y - cy
                        if abs(yaw_err) > DEAD_ZONE_YAW:
                            yaw_cmd = yaw_pid.update(yaw_err)
                        if abs(pitch_err) > DEAD_ZONE_PITCH:
                            pitch_cmd = pitch_pid.update(pitch_err)

                    cv2.putText(frame, "SEARCHING", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Altitude hold
                alt_err = TARGET_ALTITUDE - altitude
                thrust_adjustment = alt_pid.update(alt_err)
                thrust = np.clip(HOVER_THRUST + thrust_adjustment, 0.3, 0.8)

                # Send commands based on control mode
                if target_centered and best_det is not None:
                    await drone.offboard.set_velocity_ned(VelocityNedYaw(
                        forward_velocity,
                        0.0,
                        0.0,
                        np.deg2rad(yaw_cmd)
                    ))
                else:
                    await drone.offboard.set_attitude_rate(AttitudeRate(
                        0.0,
                        np.deg2rad(-pitch_cmd),
                        np.deg2rad(yaw_cmd),
                        thrust
                    ))

                # Draw UI
                cv2.line(frame, (int(cx), 0), (int(cx), h), (255, 0, 0), 1)
                cv2.line(frame, (0, int(cy)), (w, int(cy)), (255, 0, 0), 1)

                status_color = (255, 255, 0) if best_det else (0, 165, 255)
                cv2.putText(frame, f"TRACKING: {tracked_class}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(frame, f"Alt: {altitude:.1f}m (T:{TARGET_ALTITUDE:.1f}m)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # ASPS score overlay
                if target_centered:
                    center_color = (0, 255, 0)
                    mode_label = "FOLLOW"
                else:
                    center_color = (0, 165, 255)
                    mode_label = "CENTER"
                cv2.putText(frame, f"{mode_label} | LiDAR: {raw_lidar_m:.2f}m",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, center_color, 2)
                cv2.putText(frame, f"Score: {final_score:.2f}  Base:{comp_base:.2f} Conf:{comp_conf:.2f}",
                            (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, center_color, 2)
                cv2.putText(frame, f"Dist:{comp_dist:.2f} Vel:{comp_vel:.2f}  Fwd:{forward_velocity:.2f}m/s",
                            (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, center_color, 2)
                cv2.putText(frame, f"Yaw: {yaw_cmd:.1f} Pitch: {pitch_cmd:.1f}",
                            (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"dt: {dt*1000:.0f}ms", (10, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # ASPS indicator
                asps_color = (0, 255, 0) if USE_ASPS else (100, 100, 100)
                cv2.putText(frame, f"ASPS: {'ON' if USE_ASPS else 'OFF'}", (w - 140, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, asps_color, 1)

                # Centered indicator
                if target_centered:
                    cv2.circle(frame, (int(cx), int(cy)), YAW_CENTERED_THRESHOLD,
                               (0, 255, 0), 1)

                # RC status
                rc_color = (0, 255, 0) if rc_ok else (0, 0, 255)
                cv2.putText(frame, f"RC: {'OK' if rc_ok else 'LOST'}",
                            (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rc_color, 2)

            else:
                if is_offboard:
                    try:
                        await drone.offboard.stop()
                        is_offboard = False
                    except Exception:
                        pass
                yaw_pid.reset()
                pitch_pid.reset()
                alt_pid.reset()
                dist_pid.reset()
                yaw_kf.reset(cx)
                pitch_kf.reset(cy)
                dist_kf.reset(TARGET_DISTANCE_M)
                selector.reset()

                cv2.putText(frame, f"STANDBY ({mode.name})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                cv2.putText(frame, "Switch to GUIDED to resume", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Log data
            log_writer.writerow([
                time.time(), mode.name, altitude,
                TARGET_ALTITUDE - altitude,
                yaw_cmd, pitch_cmd,
                raw_lidar_m, kf_dist_m, dist_err_m, forward_velocity,
                int(target_centered),
                tracked_class, confidence, rc_ok, dt,
                comp_base, comp_conf, comp_dist, comp_vel, final_score
            ])

            if video_writer:
                video_writer.write(frame)

            try:
                display_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

            if display_should_stop.is_set():
                print("Quit signal received from display thread")
                break

    except Exception as e:
        print(f"\nError in tracking loop: {e}")

    finally:
        print("\nLanding sequence initiated...\n")
        if is_offboard:
            try:
                await drone.offboard.stop()
                print("  Offboard stopped")
            except Exception:
                pass
        try:
            print("  Landing...")
            await drone.action.land()
            await asyncio.sleep(8)
            print("  Disarming...")
            await drone.action.disarm()
            print("  Disarmed")
        except Exception as e:
            print(f"  Landing error: {e}")

        display_should_stop.set()
        try:
            display_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        display_thread.join(timeout=2.0)
        if display_thread.is_alive():
            print("Warning: Display thread did not stop cleanly")

        cap.release()
        lidar.cleanup()
        if video_writer:
            video_writer.release()
        log_file.close()

        print("\nFlight complete")
        print(f"Data saved to: {flight_dir}\n")


if __name__ == "__main__":
    try:
        asyncio.run(tracking_loop())
    except KeyboardInterrupt:
        EMERGENCY_STOP = True
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
