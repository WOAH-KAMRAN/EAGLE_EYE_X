# UAV Target Tracking System: Yaw and Pitch Alignment with PyTorch YOLO
# Optimized version using .pt models with reduced overhead
# For Jetson Orin Nano / Ubuntu PC with MAVSDK drone control

import time
import cv2
import numpy as np
import datetime
import os
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, AttitudeRate, VelocityNedYaw
from mavsdk.telemetry import FlightMode
import csv

# PyTorch and YOLO imports
try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
    print(f"PyTorch successfully imported. CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"PyTorch or Ultralytics not available: {e}")
    print("Install with: pip install torch torchvision ultralytics")
    TORCH_AVAILABLE = False

# --- Configuration ---
CONNECTION_STRING = 'udpin://127.0.0.1:14550'

# YOLO Model Configuration
YOLO_MODEL_PATH = 'best.pt'  # <--- UPDATE THIS PATH!
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

# Class priorities (higher = more important)
CLASS_PRIORITIES = {
    2: 4,  # Highest priority
    1: 3,
    3: 2,
    0: 1   # Lowest priority
}

# Control limits
YAW_LIMIT = 50
PITCH_LIMIT_ALIGN = 20
HOVER_THRUST_VALUE = 0.5
TAKEOFF_THRUST_VALUE = 0.95
TAKEOFF_VELOCITY_MPS = -0.5

# Dead zones
DEAD_ZONE_YAW = 15
DEAD_ZONE_PITCH_ALIGN = 15
ALTITUDE_CAP_M = 10.0

# PID Gains
KP_YAW = 0.5
KI_YAW = 0.01
KD_YAW = 0.2
KP_PITCH_ALIGN = 0.05
KI_PITCH_ALIGN = 0.001
KD_PITCH_ALIGN = 0.01

# Kalman Filter Gains
Q_KF = 0.1
R_KF = 10

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
USE_GSTREAMER_CAMERA = False  # Set True for RPi Cam on Jetson

GSTREAMER_PIPELINE = (
    f"nvarguscamsrc sensor_id=0 ! "
    f"video/x-raw(memory:NVMM), width=(int){CAMERA_WIDTH}, height=(int){CAMERA_HEIGHT}, "
    f"format=(string)NV12, framerate=(fraction){CAMERA_FPS}/1 ! "
    "nvvidconv flip-method=0 ! "
    f"video/x-raw, width=(int){CAMERA_WIDTH}, height=(int){CAMERA_HEIGHT}, format=(string)BGRx ! "
    "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
)

# Video Recording
RECORD_VIDEO = True
VIDEO_FOLDER = "flights1"
VIDEO_FPS = CAMERA_FPS
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')

# --- PID Controller ---
class PID:
    def __init__(self, Kp, Ki, Kd, integrator_max=500):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.I_term = 0
        self.last_error = 0
        self.integrator_max = integrator_max

    def update(self, error, dt):
        P = self.Kp * error
        self.I_term += self.Ki * error * dt
        self.I_term = np.clip(self.I_term, -self.integrator_max, self.integrator_max)
        D = self.Kd * ((error - self.last_error) / dt) if dt > 0 else 0
        self.last_error = error
        return P + self.I_term + D
    
    def reset(self):
        self.I_term = 0
        self.last_error = 0

# --- Kalman Filter ---
class KalmanFilter:
    def __init__(self, dt, Q, R, initial_pos=0.0):
        self.dt = dt
        self.Q = Q
        self.R = R
        self.x = np.array([[initial_pos], [0.0]])
        self.A = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q_matrix = np.array([[0.25*dt**4, 0.5*dt**3], [0.5*dt**3, dt**2]]) * Q
        self.R_matrix = np.array([[R]])
        self.P = np.array([[1000.0, 0.0], [0.0, 1000.0]])

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q_matrix

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R_matrix
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_estimated_state(self):
        return self.x[0, 0], self.x[1, 0]
    
    def reset(self, initial_pos=0.0):
        self.x = np.array([[initial_pos], [0.0]])
        self.P = np.array([[1000.0, 0.0], [0.0, 1000.0]])

# --- YOLO PyTorch Detector ---
class YOLODetector:
    def __init__(self, model_path):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        self.model = YOLO(model_path)
        
        # Set device
        if torch.cuda.is_available():
            self.device = 'cuda'
            print("Using CUDA for inference")
        else:
            self.device = 'cpu'
            print("Using CPU for inference")
        
        self.model.to(self.device)
        
        # Warm up model
        dummy = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)
        print("YOLO model loaded and warmed up")
    
    def detect(self, frame, conf_threshold=0.5, iou_threshold=0.4):
        """Run detection on frame and return detections"""
        results = self.model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                detections.append({
                    'xyxy': [x1, y1, x2, y2],
                    'cls': cls,
                    'conf': conf
                })
        
        return detections

# --- Drone Control Functions ---
async def arm_and_takeoff(drone, target_altitude):
    print("Waiting for drone to connect...")
    async for state in drone.telemetry.health():
        if (state.is_gyrometer_calibration_ok and 
            state.is_accelerometer_calibration_ok and 
            state.is_magnetometer_calibration_ok and 
            state.is_home_position_ok):
            print("Drone is healthy and ready")
            break
        await asyncio.sleep(1)

    print("Setting HOLD mode...")
    await asyncio.sleep(2)
    try:
        await drone.action.hold()
        await asyncio.sleep(1)
    except Exception as e:
        print(f"HOLD mode failed: {e}")

    print("Arming motors...")
    try:
        await drone.action.arm()
        await asyncio.sleep(1)
    except Exception as e:
        print(f"Arming failed: {e}")
        return False

    # Start offboard with takeoff thrust
    await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, TAKEOFF_THRUST_VALUE))
    try:
        await drone.offboard.start()
        print("Offboard mode started")
    except OffboardError as error:
        print(f"Offboard start failed: {error}")
        return False

    print(f"Taking off to {target_altitude}m...")
    takeoff_start = time.time()
    TIMEOUT = 15

    while True:
        async for position in drone.telemetry.position():
            alt = position.relative_altitude_m
            break

        print(f"Altitude: {alt:.2f}m / Target: {target_altitude:.2f}m")
        await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, TAKEOFF_THRUST_VALUE))
        
        if alt >= target_altitude * 0.95:
            print(f"Reached target altitude: {alt:.2f}m")
            break
        
        if alt >= ALTITUDE_CAP_M:
            print(f"Altitude cap {ALTITUDE_CAP_M}m reached")
            break
            
        if (time.time() - takeoff_start) > TIMEOUT:
            print(f"Takeoff timeout. Current: {alt:.2f}m")
            return False
        
        await asyncio.sleep(0.1)

    await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, HOVER_THRUST_VALUE))
    print("Hovering at target altitude")
    return True

async def send_attitude_rate_commands(drone, roll_rate, pitch_rate, yaw_rate, thrust):
    roll_rate = np.clip(roll_rate, -PITCH_LIMIT_ALIGN, PITCH_LIMIT_ALIGN)
    pitch_rate = np.clip(pitch_rate, -PITCH_LIMIT_ALIGN, PITCH_LIMIT_ALIGN)
    yaw_rate = np.clip(yaw_rate, -YAW_LIMIT, YAW_LIMIT)
    
    attitude_rate = AttitudeRate(
        np.deg2rad(roll_rate),
        np.deg2rad(-pitch_rate),
        np.deg2rad(yaw_rate),
        thrust
    )
    await drone.offboard.set_attitude_rate(attitude_rate)

# --- Main Tracking Loop ---
async def tracking_loop():
    print("Starting PyTorch YOLO tracking loop...")
    drone = System()
    await drone.connect(system_address=CONNECTION_STRING)

    print("Waiting for drone telemetry...")
    async for state in drone.telemetry.health():
        if (state.is_gyrometer_calibration_ok and 
            state.is_accelerometer_calibration_ok and 
            state.is_magnetometer_calibration_ok and 
            state.is_home_position_ok):
            break
        await asyncio.sleep(1)

    print("Waiting for OFFBOARD/GUIDED mode...")
    while True:
        async for flight_mode in drone.telemetry.flight_mode():
            if flight_mode == FlightMode.OFFBOARD:
                print("Drone ready for offboard control")
                break
            print(f"Current mode: {flight_mode.name}")
            await asyncio.sleep(1)
        if flight_mode == FlightMode.OFFBOARD:
            break

    if not await arm_and_takeoff(drone, 2.0):
        print("Takeoff failed. Exiting.")
        return

    # Initialize controllers
    yaw_pid = PID(KP_YAW, KI_YAW, KD_YAW)
    pitch_pid = PID(KP_PITCH_ALIGN, KI_PITCH_ALIGN, KD_PITCH_ALIGN)
    
    # Initialize camera
    if USE_GSTREAMER_CAMERA:
        cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return

    h, w = frame.shape[:2]
    center_x, center_y = w / 2, h / 2

    # Initialize YOLO detector
    try:
        detector = YOLODetector(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        cap.release()
        return

    # Initialize Kalman filters
    yaw_kf = KalmanFilter(dt=0.1, Q=Q_KF, R=R_KF, initial_pos=center_x)
    pitch_kf = KalmanFilter(dt=0.1, Q=Q_KF, R=R_KF, initial_pos=center_y)
    
    last_time = time.time()

    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    flight_folder = os.path.join(VIDEO_FOLDER, f"flight_{timestamp}")
    os.makedirs(flight_folder, exist_ok=True)
    
    csv_file = open(os.path.join(flight_folder, f"log_{timestamp}.csv"), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Time', 'Mode', 'Yaw_Err', 'Yaw_Cmd', 'Pitch_Err', 'Pitch_Cmd', 
                         'Class_ID', 'Altitude', 'Conf'])
    
    video_writer = None
    if RECORD_VIDEO:
        video_file = os.path.join(flight_folder, f"video_{timestamp}.mp4")
        video_writer = cv2.VideoWriter(video_file, VIDEO_CODEC, VIDEO_FPS, (w, h))

    is_offboard_active = True

    while True:
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        dt = current_time - last_time
        last_time = current_time
        
        # Predict Kalman filters
        yaw_kf.dt = dt
        yaw_kf.predict()
        pitch_kf.dt = dt
        pitch_kf.predict()

        # Get telemetry
        async for fm in drone.telemetry.flight_mode():
            current_mode = fm
            break
        
        async for pos in drone.telemetry.position():
            altitude = pos.relative_altitude_m
            break

        yaw_cmd = pitch_cmd = 0.0
        tracked_class = -1
        confidence = 0.0

        if current_mode == FlightMode.OFFBOARD:
            if not is_offboard_active:
                try:
                    await drone.offboard.start()
                    is_offboard_active = True
                except OffboardError as e:
                    print(f"Offboard restart failed: {e}")
            
            # Run detection
            detections = detector.detect(frame, CONF_THRESHOLD, IOU_THRESHOLD)
            
            # Select highest priority target
            best_det = None
            best_priority = -1
            for det in detections:
                priority = CLASS_PRIORITIES.get(det['cls'], 0)
                if priority > best_priority:
                    best_priority = priority
                    best_det = det
            
            if best_det:
                x1, y1, x2, y2 = best_det['xyxy']
                raw_x = (x1 + x2) / 2
                raw_y = (y1 + y2) / 2
                
                # Update Kalman filters
                yaw_kf.update(np.array([[raw_x]]))
                pitch_kf.update(np.array([[raw_y]]))
                
                est_x, _ = yaw_kf.get_estimated_state()
                est_y, _ = pitch_kf.get_estimated_state()
                
                yaw_err = est_x - center_x
                pitch_err = est_y - center_y
                
                # Compute commands
                if abs(yaw_err) > DEAD_ZONE_YAW:
                    yaw_cmd = yaw_pid.update(yaw_err, dt)
                else:
                    yaw_pid.reset()
                
                if abs(pitch_err) > DEAD_ZONE_PITCH_ALIGN:
                    pitch_cmd = pitch_pid.update(pitch_err, dt)
                else:
                    pitch_pid.reset()
                
                await send_attitude_rate_commands(drone, 0, pitch_cmd, yaw_cmd, HOVER_THRUST_VALUE)
                
                tracked_class = best_det['cls']
                confidence = best_det['conf']
                
                # Draw detections
                for det in detections:
                    dx1, dy1, dx2, dy2 = det['xyxy']
                    color = (0, 255, 0) if det == best_det else (0, 100, 255)
                    cv2.rectangle(frame, (int(dx1), int(dy1)), (int(dx2), int(dy2)), color, 2)
                    cv2.putText(frame, f"{det['conf']:.2f}", (int(dx1), int(dy1-5)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                cv2.circle(frame, (int(raw_x), int(raw_y)), 5, (0, 0, 255), -1)
                cv2.circle(frame, (int(est_x), int(est_y)), 8, (255, 0, 255), -1)
                cv2.putText(frame, f"Tracking ID: {tracked_class}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # No detection - use last Kalman estimate
                est_x, _ = yaw_kf.get_estimated_state()
                est_y, _ = pitch_kf.get_estimated_state()
                
                yaw_err = est_x - center_x
                pitch_err = est_y - center_y
                
                if abs(yaw_err) > DEAD_ZONE_YAW:
                    yaw_cmd = yaw_pid.update(yaw_err, dt)
                else:
                    yaw_pid.reset()
                
                if abs(pitch_err) > DEAD_ZONE_PITCH_ALIGN:
                    pitch_cmd = pitch_pid.update(pitch_err, dt)
                else:
                    pitch_pid.reset()
                
                await send_attitude_rate_commands(drone, 0, pitch_cmd, yaw_cmd, HOVER_THRUST_VALUE)
                
                cv2.circle(frame, (int(est_x), int(est_y)), 8, (255, 165, 0), -1)
                cv2.putText(frame, "No Target", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Draw center lines
            cv2.line(frame, (int(center_x), 0), (int(center_x), h), (255, 0, 0), 1)
            cv2.line(frame, (0, int(center_y)), (w, int(center_y)), (255, 0, 0), 1)
            
            cv2.putText(frame, f"Yaw: {yaw_cmd:.1f} deg/s", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Pitch: {pitch_cmd:.1f} deg/s", (10, 85), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Alt: {altitude:.2f}m", (10, 110), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        else:
            # Not in offboard mode
            if is_offboard_active:
                try:
                    await drone.offboard.stop()
                    is_offboard_active = False
                except OffboardError:
                    pass
            
            yaw_pid.reset()
            pitch_pid.reset()
            yaw_kf.reset(center_x)
            pitch_kf.reset(center_y)
            
            cv2.putText(frame, f"Mode: {current_mode.name}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Log data
        csv_writer.writerow([current_time, current_mode.name, yaw_cmd, pitch_cmd, 
                           tracked_class, altitude, confidence])
        
        if video_writer:
            video_writer.write(frame)
        
        cv2.imshow("UAV Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    print("Landing...")
    if is_offboard_active:
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass
    
    await drone.action.land()
    await asyncio.sleep(5)
    await drone.action.disarm()
    
    cap.release()
    if video_writer:
        video_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(tracking_loop())