# UAV Target Tracking System: Yaw and Pitch Alignment with TensorRT Optimization
# This script is designed for a Jetson Orin Nano / Ubuntu PC, using RPi Cam v2 (CSI) / Webcam,
# and Mission Planner for mode control.
# It uses a YOLO TensorRT engine (.engine files), PID controllers for yaw and pitch, Kalman Filters,
# custom target prioritization, MAVSDK for drone control, and includes onboard video recording.

import time
import cv2
import numpy as np
import datetime
import os
import asyncio # Required for MAVSDK
from mavsdk import System
from mavsdk.offboard import OffboardError, AttitudeRate, VelocityNedYaw
from mavsdk.telemetry import FlightMode, Position, LandedState, Health 

import csv # For saving data to CSV
import random
import threading
import queue
import atexit # For guaranteed cleanup

# --- Dependency Checks ---
TRT_AVAILABLE = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit 
    TRT_AVAILABLE = True
    print("[INIT] TensorRT and PyCUDA successfully imported.")
except ImportError:
    print("[INIT] TensorRT or PyCUDA not available. TensorRT optimization disabled.")

DEPTH_AVAILABLE = False
try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    DEPTH_AVAILABLE = True
    print("[INIT] PyTorch successfully imported for depth estimation.")
except ImportError:
    print("[INIT] PyTorch not available for depth estimation. Depth control disabled.")

SCIPY_AVAILABLE = False
try:
    from scipy.ndimage import median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    print("[INIT] Scipy not available. Depth median filter disabled.")

ULTRALYTICS_AVAILABLE = False
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("[INIT] Ultralytics not available. Real YOLO inference disabled.")

# --- Configuration ---
CONNECTION_STRING = 'udpin://127.0.0.1:14550' # <--- IMPORTANT: UPDATE THIS PATH!

# --- YOLO Model Configuration ---
USE_REAL_YOLO = False          # Set to True to use your real YOLO model, False for mock detections
REAL_YOLO_MODEL_PATH = '/path/to/your/yolov8n.engine' # <--- IMPORTANT: UPDATE THIS PATH!

# --- Depth Estimation Configuration ---
USE_DEPTH_ESTIMATION = True     
DEPTH_MODEL_TYPE = 'MiDaS_small'      
TARGET_DISTANCE_M = 2.0         
DISTANCE_DEAD_ZONE_M = 0.2     
MAX_FORWARD_SPEED_MPS = 1.0    
YAW_CENTERED_THRESHOLD = 25    

# --- Depth Map Tuning Parameters ---
DEPTH_SMOOTHING_FACTOR = 0.7       
DEPTH_ROI_SCALE = 0.3              
DEPTH_MIN_CONFIDENCE = 0.1         
DEPTH_MAX_RANGE_M = 8.0            
DEPTH_MIN_RANGE_M = 0.5            
USE_DEPTH_MEDIAN_FILTER = True and SCIPY_AVAILABLE # Conditional check
DEPTH_PERCENTILE = 25              

# Define custom priority for each class ID.
CLASS_PRIORITIES = {
    2: 4,  
    1: 3, 
    3: 2,  
    0: 1   
}

YAW_LIMIT = 50                 
PITCH_LIMIT_ALIGN = 20         
HOVER_THRUST_VALUE = 0.6       
TAKEOFF_ALTITUDE_M = 2.0
ALTITUDE_CAP_M = 10.0          

DEAD_ZONE_YAW = 15             
DEAD_ZONE_PITCH_ALIGN = 15     

# PID GAINS
KP_YAW = 0.5
KI_YAW = 0.01
KD_YAW = 0.2

KP_PITCH_ALIGN = 0.05 
KI_PITCH_ALIGN = 0.001
KD_PITCH_ALIGN = 0.01

KP_DISTANCE = 1.0     
KI_DISTANCE = 0.02    
KD_DISTANCE = 0.25    

# KALMAN FILTER GAINS
Q_KF = 0.1
R_KF = 10
Q_DISTANCE_KF = 0.15 
R_DISTANCE_KF = 0.8  

# --- Camera Configuration ---
CAMERA_WIDTH = 640  
CAMERA_HEIGHT = 480 
CAMERA_FPS = 30     

GSTREAMER_PIPELINE = (
    f"nvarguscamsrc sensor_id=0 ! "
    f"video/x-raw(memory:NVMM), width=(int){CAMERA_WIDTH}, height=(int){CAMERA_HEIGHT}, format=(string)NV12, framerate=(fraction){CAMERA_FPS}/1 ! "
    "nvvidconv flip-method=0 ! "
    f"video/x-raw, width=(int){CAMERA_WIDTH}, height=(int){CAMERA_HEIGHT}, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
)

# --- Camera Selection Flag ---
USE_GSTREAMER_CAMERA = False 

# --- Video Recording Configuration ---
RECORD_VIDEO = True
VIDEO_FOLDER = "flights"
VIDEO_FPS = CAMERA_FPS 
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v') 
display_queue = queue.Queue(maxsize=2)

# Flag to signal display thread to stop
display_should_stop = threading.Event()

# Global placeholders for resources
drone = None
cap = None
video_writer = None
csv_file = None
display_thread = None

# --- PID Controller Class ---
class PID:
    """A basic PID controller for yaw or pitch control."""
    def __init__(self, Kp, Ki, Kd, integrator_max=500):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.P_term = 0
        self.I_term = 0
        self.D_term = 0
        self.last_error = 0
        self.integrator_max = integrator_max

    def update(self, error, dt):
        """Calculates PID output based on error and time delta."""
        self.P_term = self.Kp * error
        self.I_term += self.Ki * error * dt
        self.I_term = np.clip(self.I_term, -self.integrator_max, self.integrator_max)
        self.D_term = self.Kd * ((error - self.last_error) / dt) if dt > 0 else 0
        self.last_error = error
        return self.P_term + self.I_term + self.D_term
    
    def reset(self):
        """Resets the PID controller's state."""
        self.P_term = 0
        self.I_term = 0
        self.D_term = 0
        self.last_error = 0

# --- Kalman Filter Class ---
class KalmanFilter:
    """A simple 1D Kalman Filter for estimating a single state."""
    def __init__(self, dt, Q, R, initial_pos=0.0):
        self.dt = dt
        self.Q = Q 
        self.R = R 
        self.x = np.array([[initial_pos], [0.0]]) # [position, velocity]
        self.A = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q_matrix = np.array([[0.25*dt**4, 0.5*dt**3], [0.5*dt**3, dt**2]]) * self.Q
        self.R_matrix = np.array([[self.R]])
        self.P = np.array([[1000.0, 0.0], [0.0, 1000.0]])

    def predict(self):
        """Predicts the next state."""
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q_matrix

    def update(self, z):
        """Updates the state with a new measurement."""
        y = z - np.dot(self.H, self.x) 
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R_matrix 
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) 
        
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_estimated_state(self):
        """Returns the estimated position and velocity."""
        return self.x[0, 0], self.x[1, 0] 
    
    def reset(self, initial_pos=0.0):
        """Resets the Kalman filter's state."""
        self.x = np.array([[initial_pos], [0.0]])
        self.P = np.array([[1000.0, 0.0], [0.0, 1000.0]]) 
    
    def update_dt(self, dt):
        """Update the time step and recalculate matrices."""
        self.dt = dt
        self.A = np.array([[1, dt], [0, 1]])
        self.Q_matrix = np.array([[0.25*dt**4, 0.5*dt**3],
                                  [0.5*dt**3, dt**2]]) * self.Q

# --- Distance Kalman Filter (specialized for depth estimation) ---
class DistanceKalmanFilter(KalmanFilter):
    """Specialized Kalman Filter for distance estimation."""
    def __init__(self, dt, Q, R, initial_distance=5.0):
        super().__init__(dt, Q, R, initial_pos=initial_distance)
        self.P = np.array([[100.0, 0.0], [0.0, 100.0]]) # Reduced initial uncertainty

# --- Monocular Depth Estimator Class ---
class MonocularDepthEstimator:
    """Monocular depth estimation using MiDaS model for target distance calculation."""
    def __init__(self, model_type='MiDaS_small'):
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_distance = None # For temporal smoothing
        
    def load_model(self):
        """Load the MiDaS depth estimation model."""
        if not DEPTH_AVAILABLE:
            return False
            
        try:
            if self.model_type == 'MiDaS_small':
                self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
                self.model.to(self.device)
                self.model.eval()
                
                midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
                self.transform = midas_transforms.small_transform
                return True
            else:
                print(f"[DEPTH] Unsupported depth model type: {self.model_type}")
                return False
                
        except Exception as e:
            print(f"[DEPTH] Failed to load depth model: {e}")
            return False
    
    def estimate_depth(self, frame):
        """Estimate depth map from input frame."""
        if self.model is None:
            return None
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(rgb_frame).to(self.device)
            
            with torch.no_grad():
                depth_map = self.model(input_tensor)
                
            depth_map = depth_map.squeeze().cpu().numpy()
            return depth_map
            
        except Exception as e:
            print(f"[DEPTH] Estimation error: {e}")
            return None
    
    def get_target_distance(self, frame, target_bbox, image_center_x, image_center_y):
        """Get estimated distance to target center in meters with tunable parameters."""
        depth_map = self.estimate_depth(frame)
        if depth_map is None:
            return None
            
        try:
            x1, y1, x2, y2 = target_bbox
            
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Define ROI around the center of the detection
            scaled_width = bbox_width * DEPTH_ROI_SCALE
            scaled_height = bbox_height * DEPTH_ROI_SCALE
            
            roi_x1 = int(center_x - scaled_width / 2)
            roi_y1 = int(center_y - scaled_height / 2)
            roi_x2 = int(center_x + scaled_width / 2)
            roi_y2 = int(center_y + scaled_height / 2)
            
            h, w = depth_map.shape
            frame_h, frame_w = frame.shape[:2]
            
            # Map ROI coordinates to depth map coordinates
            depth_x1 = max(0, int(roi_x1 * w / frame_w))
            depth_y1 = max(0, int(roi_y1 * h / frame_h))
            depth_x2 = min(w, int(roi_x2 * w / frame_w))
            depth_y2 = min(h, int(roi_y2 * h / frame_h))
            
            if depth_x2 <= depth_x1 or depth_y2 <= depth_y1:
                return None
            
            depth_samples = depth_map[depth_y1:depth_y2, depth_x1:depth_x2]
            
            # Apply median filter if enabled and scipy is available
            if USE_DEPTH_MEDIAN_FILTER and depth_samples.size > 9:
                depth_samples = median_filter(depth_samples, size=3)
            
            if depth_samples.size > 0:
                # Use percentile for robust depth estimation (closer objects)
                target_depth = np.percentile(depth_samples.flatten(), DEPTH_PERCENTILE)
            else:
                return None
            
            # MiDaS outputs inverse depth. Adjust conversion factor.
            if target_depth > 0:
                estimated_distance = 8.0 / target_depth 
                
                # Apply range limits
                estimated_distance = max(DEPTH_MIN_RANGE_M, min(DEPTH_MAX_RANGE_M, estimated_distance))
                
                # Apply temporal smoothing
                if self.last_distance is not None:
                    estimated_distance = (DEPTH_SMOOTHING_FACTOR * self.last_distance + 
                                        (1 - DEPTH_SMOOTHING_FACTOR) * estimated_distance)
                
                # Simple confidence check based on depth variance
                depth_std = np.std(depth_samples)
                confidence = 1.0 / (1.0 + depth_std)
                
                if confidence >= DEPTH_MIN_CONFIDENCE:
                    self.last_distance = estimated_distance
                    return estimated_distance
                else:
                    return None 
            else:
                return None
                
        except Exception as e:
            print(f"[DEPTH] Error getting target distance: {e}")
            return None

    def create_depth_visualization(self, depth_map, frame_shape):
        """Create a colorized depth map for visualization."""
        if depth_map is None:
            return None
            
        try:
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
            depth_resized = cv2.resize(depth_colored, (frame_shape[1], frame_shape[0]))
            return depth_resized
        except Exception as e:
            return None

# --- Mock Target Movement ---
mock_target_pos = { 0: {'x': 0, 'y': 0}, 1: {'x': 0, 'y': 0}, 2: {'x': 0, 'y': 0}, 3: {'x': 0, 'y': 0} }

def initialize_mock_targets(width, height):
    """Initializes mock target positions based on image dimensions."""
    global mock_target_pos
    mock_target_pos[0] = {'x': width * 0.2, 'y': height * 0.3}
    mock_target_pos[1] = {'x': width * 0.5, 'y': height * 0.5}
    mock_target_pos[2] = {'x': width * 0.8, 'y': height * 0.7} # Priority 2
    mock_target_pos[3] = {'x': width * 0.3, 'y': height * 0.6}

def mock_yolo_inference(frame):
    """Simulate YOLOv8s object detection with movement and noise."""
    height, width, _ = frame.shape
    results = []

    for class_id in mock_target_pos:
        # Simulate simple wandering movement
        mock_target_pos[class_id]['x'] += random.uniform(-5, 5) 
        mock_target_pos[class_id]['y'] += random.uniform(-3, 3) 
        mock_target_pos[class_id]['x'] = np.clip(mock_target_pos[class_id]['x'], width * 0.1, width * 0.9)
        mock_target_pos[class_id]['y'] = np.clip(mock_target_pos[class_id]['y'], height * 0.1, height * 0.9)

    def create_mock_detection(x_center_m, y_center_m, class_id):
        detected_x_center = x_center_m + random.uniform(-10, 10)
        detected_y_center = y_center_m + random.uniform(-5, 5)
        box_width = 80 + random.uniform(-10, 10)
        box_height = 80 + random.uniform(-10, 10)

        x1 = detected_x_center - box_width / 2
        y1 = detected_y_center - box_height / 2
        x2 = detected_x_center + box_width / 2
        y2 = detected_y_center + box_height / 2
        
        return {
            'xyxy': [x1, y1, x2, y2],
            'cls': class_id
        }

    # Simulate presence of multiple classes (prioritized: 2 > 1 > 3 > 0)
    if random.random() < 0.9: 
        results.append(create_mock_detection(mock_target_pos[2]['x'], mock_target_pos[2]['y'], 2))
    
    if random.random() < 0.7: 
        results.append(create_mock_detection(mock_target_pos[1]['x'], mock_target_pos[1]['y'], 1))

    if random.random() < 0.6: 
        results.append(create_mock_detection(mock_target_pos[3]['x'], mock_target_pos[3]['y'], 3))
    
    if random.random() < 0.5: 
        results.append(create_mock_detection(mock_target_pos[0]['x'], mock_target_pos[0]['y'], 0))

    return results

# --- Drone Control Functions (MAVSDK) ---
async def arm_and_takeoff(drone, aTargetAltitude):
    """Arms vehicle and flies to aTargetAltitude using MAVSDK offboard commands."""
    print("[DRONE] Waiting for drone to connect...")
    async for state in drone.telemetry.health():
        if state.is_gyrometer_calibration_ok and state.is_accelerometer_calibration_ok and state.is_magnetometer_calibration_ok and state.is_home_position_ok:
            print("[DRONE] Drone is healthy and ready to arm.")
            break
        await asyncio.sleep(1)
        
    print("[DRONE] Arming motors...")
    try:
        await drone.action.arm()
        print("[DRONE] Motors armed successfully.")
        await asyncio.sleep(1)
    except Exception as e:
        print(f"[DRONE] Arming failed with error: {e}")
        return False

    # Start offboard mode by sending a neutral attitude rate with takeoff thrust
    await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, HOVER_THRUST_VALUE + 0.3))
    try:
        await drone.offboard.start()
        print(f"[DRONE] Offboard mode started for takeoff.")
    except OffboardError as error:
        print(f"[DRONE] Starting offboard mode failed: {error}. Disarming.")
        await drone.action.disarm()
        return False

    print(f"[DRONE] Taking off to {aTargetAltitude} meters...")
    takeoff_start_time = time.time()
    TAKE_OFF_TIMEOUT = 20

    while True:
        current_altitude = 0.0
        try:
            async for position in drone.telemetry.position():
                current_altitude = position.relative_altitude_m
                break
        except Exception:
             # Handle telemetry stream errors gracefully
             print("[DRONE] Telemetry read error during takeoff.")
             await asyncio.sleep(0.5)
             continue

        print(f"[DRONE] Altitude: {current_altitude:.2f}m / Target: {aTargetAltitude:.2f}m")
        # Keep sending takeoff thrust
        await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, HOVER_THRUST_VALUE + 0.3)) 
        
        if current_altitude >= aTargetAltitude * 0.95:
            print(f"[DRONE] Reached target altitude: {current_altitude:.2f}m. Switching to hover thrust.")
            # Set to hover thrust after reaching altitude
            await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, HOVER_THRUST_VALUE))
            return True
            
        if (time.time() - takeoff_start_time) > TAKE_OFF_TIMEOUT:
            print(f"[DRONE] Takeoff timed out. Current altitude: {current_altitude:.2f}m")
            return False 
            
        await asyncio.sleep(0.1) 

async def send_attitude_rate_commands(drone, roll_rate, pitch_rate, yaw_rate, thrust_value):
    """Sends roll, pitch, and yaw rate commands to the drone using MAVSDK offboard."""
    roll_rate = np.clip(roll_rate, -PITCH_LIMIT_ALIGN, PITCH_LIMIT_ALIGN)
    pitch_rate = np.clip(pitch_rate, -PITCH_LIMIT_ALIGN, PITCH_LIMIT_ALIGN)
    yaw_rate = np.clip(yaw_rate, -YAW_LIMIT, YAW_LIMIT)

    # Note on Pitch Inversion: Positive pitch rate (MAVSDK) = nose up.
    # If error_align is positive (target below center), we want nose down (negative pitch rate).
    # Thus, we send -pitch_rate.
    attitude_rate = AttitudeRate(
        np.deg2rad(roll_rate),
        np.deg2rad(-pitch_rate), 
        np.deg2rad(yaw_rate),
        thrust_value # Thrust from 0.0 to 1.0
    )
    try:
        await drone.offboard.set_attitude_rate(attitude_rate)
    except OffboardError as e:
        print(f"[DRONE] Warning: Failed to send attitude rate command: {e}")
    except Exception as e:
        print(f"[DRONE] Warning: Generic error sending attitude rate command: {e}")

# --- Display Thread Function ---
def display_thread_func():
    """Separate thread for displaying frames."""
    print("[DISPLAY] Starting display thread...")
    cv2.namedWindow("Object Tracking", cv2.WINDOW_NORMAL)
    
    while not display_should_stop.is_set():
        try:
            frame = display_queue.get(timeout=0.1)
            
            if frame is None:
                break
            
            cv2.imshow("Object Tracking", frame)
            
            # Key press check in non-blocking wait
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("[DISPLAY] 'q' pressed, signaling main loop to stop")
                display_should_stop.set()
                break
                
        except queue.Empty:
            cv2.waitKey(1)
            continue
        
        except Exception as e:
            print(f"[DISPLAY] Error in display thread: {e}")
            break
    
    print("[DISPLAY] Closing window...")
    cv2.destroyAllWindows()
    print("[DISPLAY] Stopped")

def initialize_camera(width, height, fps):
    """Initializes and returns the camera capture object."""
    if USE_GSTREAMER_CAMERA:
        cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
        print("[CAMERA] Attempting to initialize GStreamer pipeline.")
    else:
        # Try different indices for standard webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
             cap = cv2.VideoCapture(1)
        print("[CAMERA] Attempting to initialize standard OpenCV camera.")
             
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Log actual values obtained by the camera for debugging
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[CAMERA] Requested: {width}x{height}@{fps} FPS. Actual: {actual_width}x{actual_height}@{actual_fps:.1f} FPS.")
        
    return cap

def cleanup():
    """Performs synchronous cleanup of resources."""
    global cap, video_writer, csv_file, display_thread, drone

    print("\n[CLEANUP] Initiating synchronous cleanup...")

    # 1. Signal and stop display thread
    if display_thread and display_thread.is_alive():
        print("[CLEANUP] Stopping display thread...")
        display_should_stop.set()
        try:
            # Send sentinel value
            display_queue.put(None, timeout=0.5) 
        except queue.Full:
            pass
        display_thread.join(timeout=2.0)
        if display_thread.is_alive():
            print("[CLEANUP] Warning: Display thread did not stop cleanly.")
    
    # 2. Release camera
    if cap:
        print("[CLEANUP] Releasing camera.")
        cap.release()
        
    # 3. Release video writer
    if video_writer:
        print("[CLEANUP] Releasing video writer.")
        video_writer.release()
        
    # 4. Close CSV file
    if csv_file:
        print("[CLEANUP] Closing CSV file.")
        csv_file.close()

async def tracking_loop():
    global drone, cap, video_writer, csv_file, display_thread
    
    print("[MAIN] Starting tracking loop...")
    
    # --- Resource Initialization ---
    drone = System()
    
    # Start display thread first
    display_thread = threading.Thread(target=display_thread_func, daemon=True, name="display_thread")
    display_thread.start()
    await asyncio.sleep(0.5) # Give the thread time to start

    try:
        await drone.connect(system_address=CONNECTION_STRING)
        print(f"[MAIN] MAVSDK connecting to {CONNECTION_STRING}...")
        
        # Camera Setup
        cap = initialize_camera(CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        if not cap.isOpened():
            print("[ERROR] Could not open camera. Exiting.")
            return

        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read initial frame. Exiting.")
            return

        image_height, image_width, _ = frame.shape
        image_center_x = image_width / 2
        image_center_y = image_height / 2 
        initialize_mock_targets(image_width, image_height)
        
        # Model Loading
        yolo_model = None
        _use_real_yolo_local = USE_REAL_YOLO and ULTRALYTICS_AVAILABLE
        if _use_real_yolo_local:
            try:
                yolo_model = YOLO(REAL_YOLO_MODEL_PATH)
                print(f"[YOLO] Real YOLO model loaded from: {REAL_YOLO_MODEL_PATH}")
            except Exception as e:
                print(f"[YOLO] Error loading real YOLO model: {e}. Falling back to mock.")
                _use_real_yolo_local = False

        depth_estimator = None
        _use_depth_local = USE_DEPTH_ESTIMATION and DEPTH_AVAILABLE
        if _use_depth_local:
            try:
                depth_estimator = MonocularDepthEstimator(DEPTH_MODEL_TYPE)
                if not depth_estimator.load_model():
                    _use_depth_local = False
                    depth_estimator = None
            except Exception as e:
                print(f"[DEPTH] Error initializing depth estimator: {e}. Disabled.")
                _use_depth_local = False
                depth_estimator = None

        # Drone Takeoff
        takeoff_successful = await arm_and_takeoff(drone, TAKEOFF_ALTITUDE_M) 
        if not takeoff_successful:
            print("[MAIN] Takeoff failed. Exiting tracking loop.")
            return 

        # Controller Initialization
        yaw_pid_controller = PID(KP_YAW, KI_YAW, KD_YAW)
        pitch_align_pid_controller = PID(KP_PITCH_ALIGN, KI_PITCH_ALIGN, KD_PITCH_ALIGN)
        distance_pid_controller = PID(KP_DISTANCE, KI_DISTANCE, KD_DISTANCE)
        yaw_kalman_filter = KalmanFilter(dt=0.1, Q=Q_KF, R=R_KF, initial_pos=image_center_x) 
        pitch_kalman_filter = KalmanFilter(dt=0.1, Q=Q_KF, R=R_KF, initial_pos=image_center_y) 
        distance_kalman_filter = DistanceKalmanFilter(dt=0.1, Q=Q_DISTANCE_KF, R=R_DISTANCE_KF, initial_distance=TARGET_DISTANCE_M)
        
        # Logging Setup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        flight_folder = os.path.join(VIDEO_FOLDER, f"flight_data_{timestamp}")
        os.makedirs(flight_folder, exist_ok=True)
        data_filename = os.path.join(flight_folder, f"flight_log_{timestamp}.csv")

        csv_headers = [
            'Time_s', 'FlightMode', 
            'Raw_Yaw_Err_px', 'KF_Yaw_Est_px', 'Yaw_Cmd_deg_s', 'Yaw_I_term',
            'Raw_Pitch_Align_Err_px', 'KF_Pitch_Align_Est_px', 'Pitch_Cmd_deg_s', 'Pitch_I_term',
            'Target_Distance_m', 'KF_Distance_Est_m', 'Distance_Err_m', 'Forward_Velocity_mps',
            'Target_Centered', 'Target_Class_ID'
        ]
        csv_file = open(data_filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_headers)
        
        # Video Setup
        if RECORD_VIDEO:
            video_filename = os.path.join(flight_folder, f"tracking_video_{timestamp}.mp4")
            video_writer = cv2.VideoWriter(video_filename, VIDEO_CODEC, VIDEO_FPS, (image_width, image_height))
            if not video_writer.isOpened():
                print(f"[VIDEO] Error: Could not open video writer for {video_filename}. Recording disabled.")
                video_writer = None

        last_time = time.time()
        is_offboard_active_from_script = True # True after successful takeoff

        # --- Main Tracking Loop ---
        while not display_should_stop.is_set():
            current_time = time.time() 
            dt = current_time - last_time
            last_time = current_time

            # Handle camera frame read
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            # Update Kalman Filter Time Steps
            yaw_kalman_filter.update_dt(dt)
            pitch_kalman_filter.update_dt(dt)
            distance_kalman_filter.update_dt(dt)
            yaw_kalman_filter.predict()
            pitch_kalman_filter.predict()
            distance_kalman_filter.predict()

            yaw_rate_cmd = 0.0
            pitch_rate_cmd = 0.0
            forward_velocity_cmd = 0.0
            
            # Telemetry read (non-blocking)
            current_flight_mode = FlightMode.UNKNOWN
            try:
                async for flight_mode_telemetry in drone.telemetry.flight_mode():
                    current_flight_mode = flight_mode_telemetry
                    break
            except Exception:
                pass # Ignore telemetry errors, continue with last known mode (or UNKNOWN)

            log_data_row = [current_time, current_flight_mode.name] 

            if current_flight_mode == FlightMode.OFFBOARD:
                
                # --- YOLO Inference ---
                detections = []
                if _use_real_yolo_local and yolo_model:
                    try:
                        results = yolo_model.predict(frame, verbose=False, conf=0.5) 
                        for r in results:
                            for box in r.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                cls_id = int(box.cls[0])
                                detections.append({
                                    'xyxy': [x1, y1, x2, y2],
                                    'cls': cls_id
                                })
                    except Exception as e:
                        print(f"[YOLO] Real YOLO inference error: {e}. Using mock detections.")
                        detections = mock_yolo_inference(frame) 
                else:
                    detections = mock_yolo_inference(frame)
                
                # --- Target Prioritization ---
                selected_box = None
                highest_priority_score = -1
                for detection in detections:
                    priority_score = CLASS_PRIORITIES.get(detection['cls'], -1) 
                    if priority_score > highest_priority_score:
                        highest_priority_score = priority_score
                        selected_box = detection
                
                # --- Control Initialization ---
                raw_yaw_err = 0.0
                raw_pitch_align_err = 0.0
                target_distance_m = 0.0
                distance_err = 0.0
                target_centered = False
                tracked_class_id = -1
                
                estimated_x_center, _ = yaw_kalman_filter.get_estimated_state()
                estimated_y_center, _ = pitch_kalman_filter.get_estimated_state()

                if selected_box is not None:
                    x1, y1, x2, y2 = selected_box['xyxy']
                    raw_x_center_bbox = (x1 + x2) / 2
                    raw_y_center_bbox = (y1 + y2) / 2 
                    tracked_class_id = selected_box['cls']
                    
                    # Update Kalman Filters with new measurement
                    yaw_kalman_filter.update(np.array([[raw_x_center_bbox]]))
                    pitch_kalman_filter.update(np.array([[raw_y_center_bbox]]))
                    estimated_x_center, _ = yaw_kalman_filter.get_estimated_state()
                    estimated_y_center, _ = pitch_kalman_filter.get_estimated_state()

                    yaw_error = estimated_x_center - image_center_x
                    pitch_error_align = estimated_y_center - image_center_y
                    
                    # 1. Yaw and Vertical Pitch Control (Alignment)
                    if abs(yaw_error) > DEAD_ZONE_YAW:
                        yaw_rate_cmd = yaw_pid_controller.update(yaw_error, dt)
                    else:
                        yaw_rate_cmd = 0.0
                        yaw_pid_controller.reset()

                    if abs(pitch_error_align) > DEAD_ZONE_PITCH_ALIGN:
                        pitch_rate_cmd = pitch_align_pid_controller.update(pitch_error_align, dt)
                    else:
                        pitch_rate_cmd = 0.0
                        pitch_align_pid_controller.reset()

                    # 2. Distance Control (Forward Movement)
                    target_centered = abs(yaw_error) <= YAW_CENTERED_THRESHOLD
                    
                    if target_centered and _use_depth_local and depth_estimator:
                        target_distance_m = depth_estimator.get_target_distance(
                            frame, selected_box['xyxy'], image_center_x, image_center_y
                        )
                        
                        if target_distance_m is not None:
                            distance_kalman_filter.update(np.array([[target_distance_m]]))
                            estimated_distance, _ = distance_kalman_filter.get_estimated_state()
                            
                            distance_err = estimated_distance - TARGET_DISTANCE_M
                            
                            if abs(distance_err) > DISTANCE_DEAD_ZONE_M:
                                forward_velocity_cmd = distance_pid_controller.update(distance_err, dt)
                                forward_velocity_cmd = np.clip(forward_velocity_cmd, -MAX_FORWARD_SPEED_MPS, MAX_FORWARD_SPEED_MPS)
                            else:
                                forward_velocity_cmd = 0.0
                                distance_pid_controller.reset()
                            
                            # Send Velocity Command (North/Forward and Yaw Rate)
                            await drone.offboard.set_velocity_ned(VelocityNedYaw(
                                forward_velocity_cmd, 
                                0.0, 
                                0.0, 
                                np.deg2rad(yaw_rate_cmd) 
                            ))
                        else:
                            # Depth failed: fall back to Attitude Rate for alignment
                            forward_velocity_cmd = 0.0
                            await send_attitude_rate_commands(drone, 0, pitch_rate_cmd, yaw_rate_cmd, HOVER_THRUST_VALUE)
                    else:
                        # Not centered or depth disabled: focus on alignment with Attitude Rate
                        forward_velocity_cmd = 0.0
                        await send_attitude_rate_commands(drone, 0, pitch_rate_cmd, yaw_rate_cmd, HOVER_THRUST_VALUE)
                    
                    # Log Update (Target Found)
                    raw_yaw_err = raw_x_center_bbox - image_center_x
                    raw_pitch_align_err = raw_y_center_bbox - image_center_y

                    # --- Visualization (Target Found) ---
                    for detection in detections:
                        dx1, dy1, dx2, dy2 = detection['xyxy']
                        d_class_id = detection['cls']
                        color = (0, 100, 255) if detection != selected_box else (0, 255, 0)
                        cv2.rectangle(frame, (int(dx1), int(dy1)), (int(dx2), int(dy2)), color, 2)
                        cv2.putText(frame, f"ID:{d_class_id}", (int(dx1), int(dy1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    cv2.circle(frame, (int(raw_x_center_bbox), int(raw_y_center_bbox)), 5, (0, 0, 255), -1) 
                    
                else: # No Target Found
                    # Continue using KF prediction and command rates
                    yaw_error = estimated_x_center - image_center_x
                    pitch_error_align = estimated_y_center - image_center_y

                    if abs(yaw_error) > DEAD_ZONE_YAW:
                        yaw_rate_cmd = yaw_pid_controller.update(yaw_error, dt)
                    else:
                        yaw_rate_cmd = 0.0
                        yaw_pid_controller.reset()

                    if abs(pitch_error_align) > DEAD_ZONE_PITCH_ALIGN:
                        pitch_rate_cmd = pitch_align_pid_controller.update(pitch_error_align, dt)
                    else:
                        pitch_rate_cmd = 0.0
                        pitch_align_pid_controller.reset()
                    
                    await send_attitude_rate_commands(drone, 0, pitch_rate_cmd, yaw_rate_cmd, HOVER_THRUST_VALUE) 

                    # Log Update (No Target)
                    raw_yaw_err = 0.0
                    raw_pitch_align_err = 0.0
                    target_distance_m = 0.0
                    distance_err = 0.0
                    forward_velocity_cmd = 0.0
                    
                    # --- Visualization (No Target) ---
                    cv2.circle(frame, (int(estimated_x_center), int(estimated_y_center)), 8, (255, 165, 0), -1) 
                    cv2.putText(frame, "No Target Detected - Predicting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


                # --- Common Visualization ---
                kf_distance_est, _ = distance_kalman_filter.get_estimated_state()
                kf_yaw_est, _ = yaw_kalman_filter.get_estimated_state()
                kf_pitch_align_est, _ = pitch_kalman_filter.get_estimated_state()
                
                cv2.circle(frame, (int(kf_yaw_est), int(kf_pitch_align_est)), 8, (255, 0, 255), 2) # Estimated center (magenta circle outline)
                cv2.line(frame, (int(image_center_x), 0), (int(image_center_x), image_height), (255, 0, 0), 2) 
                cv2.line(frame, (0, int(image_center_y)), (image_width, int(image_center_y)), (255, 0, 0), 2) 

                cv2.putText(frame, "Tracking: ON", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw Err: {yaw_error:.2f}px | Cmd: {yaw_rate_cmd:.1f}deg/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Pitch Align Err: {pitch_error_align:.2f}px | Cmd: {pitch_rate_cmd:.1f}deg/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if target_centered and selected_box is not None:
                    status_text = "CENTERED - Moving Forward" if forward_velocity_cmd != 0 else "CENTERED - At Target Distance"
                    status_color = (0, 255, 0)
                    cv2.putText(frame, status_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    cv2.putText(frame, f"Distance: {kf_distance_est:.1f}m | Target: {TARGET_DISTANCE_M:.1f}m", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Forward Vel: {forward_velocity_cmd:.2f}m/s", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


                # --- CSV Logging (Final Values) ---
                log_data_row.extend([
                    raw_yaw_err, kf_yaw_est, yaw_rate_cmd, yaw_pid_controller.I_term,
                    raw_pitch_align_err, kf_pitch_align_est, pitch_rate_cmd, pitch_align_pid_controller.I_term,
                    target_distance_m, kf_distance_est, distance_err, forward_velocity_cmd,
                    target_centered, tracked_class_id
                ])
                csv_writer.writerow(log_data_row)
                
            else: # Not in OFFBOARD mode
                if is_offboard_active_from_script:
                    try:
                        await drone.offboard.stop()
                        print("[DRONE] Offboard mode stopped due to mode change.")
                        is_offboard_active_from_script = False
                    except OffboardError:
                        pass # Ignore if it failed to stop, likely already stopped
                
                # Reset controllers and send neutral commands
                await send_attitude_rate_commands(drone, 0, 0, 0, HOVER_THRUST_VALUE) 
                yaw_pid_controller.reset()
                pitch_align_pid_controller.reset()
                distance_pid_controller.reset()
                
                # Log default values
                log_data_row.extend([0.0, image_center_x, 0.0, 0.0, 0.0, image_center_y, 0.0, 0.0, 0.0, TARGET_DISTANCE_M, 0.0, 0.0, False, -1])
                csv_writer.writerow(log_data_row)
                
                cv2.putText(frame, f"Tracking: OFF ({current_flight_mode.name} mode)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            # --- Video Recording and Display ---
            if video_writer is not None:
                video_writer.write(frame)
            
            # --- FRAME QUEUE OPTIMIZATION: Drop oldest frame if queue is full ---
            try:
                # Discard the oldest frame if the queue is full before putting the new one
                if display_queue.full():
                    display_queue.get_nowait()
                display_queue.put_nowait(frame.copy())
            except (queue.Full, queue.Empty):
                # Handle race conditions gracefully
                pass
            
            await asyncio.sleep(0.001)

    except Exception as e:
        print(f"[ERROR] Critical error in tracking loop: {e}")
    
    finally:
        print("[MAIN] Tracking loop finished. Attempting graceful shutdown of drone.")
        
        if drone:
            # 1. Stop Offboard Mode
            if is_offboard_active_from_script:
                try:
                    await drone.offboard.stop()
                    print("[DRONE] Offboard mode stopped.")
                except OffboardError:
                    print("[DRONE] Warning: Offboard already stopped or failed to stop.")

            # 2. Land the Drone
            try:
                print("[DRONE] Initiating land command...")
                await drone.action.land()
                # Wait for landing (optional, but good practice)
                async for landed_state in drone.telemetry.landed_state():
                    if landed_state == LandedState.ON_GROUND:
                        print("[DRONE] Landed successfully.")
                        break
                    await asyncio.sleep(1)
            except Exception as e:
                print(f"[DRONE] Warning: Land command failed: {e}")

            # 3. Disarm
            try:
                await drone.action.disarm()
                print("[DRONE] Disarmed.")
            except Exception as e:
                print(f"[DRONE] Warning: Disarm command failed: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Register synchronous cleanup to run on exit or Ctrl+C
    atexit.register(cleanup)
    
    try:
        # Start the asynchronous tracking loop
        asyncio.run(tracking_loop())
    except KeyboardInterrupt:
        print("\n[MAIN] Caught KeyboardInterrupt. Shutting down...")
    except Exception as e:
        print(f"\n[MAIN] Unhandled system error: {e}")
    finally:
        # Since asyncio.run handles thread cleanup, atexit will handle the rest.
        pass
