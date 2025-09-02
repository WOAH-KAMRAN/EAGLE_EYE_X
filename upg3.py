# UAV Target Tracking System: Yaw and Pitch Alignment (No Depth-based Distance Following)
# This script is designed for a Jetson Orin Nano / Ubuntu PC, using RPi Cam v2 (CSI) / Webcam,
# and Mission Planner for mode control.
# It uses a YOLO ONNX model placeholder, PID controllers for yaw and pitch, Kalman Filters,
# custom target prioritization, MAVSDK for drone control, and includes onboard video recording.

import time
import cv2
import numpy as np
import datetime
import os
import asyncio # Required for MAVSDK
from mavsdk import System
from mavsdk.offboard import OffboardError, AttitudeRate, VelocityNedYaw
from mavsdk.telemetry import FlightMode, Position, LandedState, Health # Import Health for clarity

import csv # For saving data to CSV
import random
import onnxruntime as ort # For ONNX model inference

# --- Configuration ---
# MAVSDK connection string. For SITL on the same PC, use udpin://:14550
# For real drone via USB/UART on Jetson, e.g., 'serial:///dev/ttyTHS1:921600'
CONNECTION_STRING = 'udpin://127.0.0.1:14550'

# --- YOLO Model Configuration ---
USE_REAL_YOLO = False          # Set to True to use your real YOLO model, False for mock detections
# Path to your ONNX model or TensorRT engine file (e.g., 'yolov8n.engine' or 'yolov8n.onnx')
REAL_YOLO_MODEL_PATH = '/path/to/your/yolov8n.engine' # <--- IMPORTANT: UPDATE THIS PATH!
# If using ONNX, ensure 'onnxruntime-gpu' is installed: pip install onnxruntime-gpu
# If using TensorRT engine, ensure it's generated from your ONNX model on the Jetson.

# Define custom priority for each class ID. Higher value means higher priority.
# Priority: Class 2 > Class 1 > Class 3 > Class 0
CLASS_PRIORITIES = {
    2: 4,  # Highest priority
    1: 3,  # Second highest
    3: 2,  # Third highest
    0: 1   # Lowest priority
}
ALTITUDE_CAP_M = 10.0  # or whatever maximum altitude you want
YAW_LIMIT = 50          # Maximum yaw speed in degrees/second
PITCH_LIMIT_ALIGN = 20         # Maximum pitch speed in degrees/second for vertical alignment
HOVER_THRUST_VALUE = 0.5      # Thrust value for hovering (0.0 to 1.0) - Tune this for stable hover
TAKEOFF_VELOCITY_MPS = -0.1  # Takeoff climb velocity in m/s (negative for upward) - INCREASED FOR RELIABLE LIFT
TAKEOFF_THRUST_VALUE = 0.8

DEAD_ZONE_YAW = 15             # Dead zone in pixels for yaw control
DEAD_ZONE_PITCH_ALIGN = 15     # Dead zone in pixels for vertical pitch alignment

# PID GAINS (tune these for yaw!)
KP_YAW = 0.5
KI_YAW = 0.01
KD_YAW = 0.2

# PID GAINS (tune these for vertical pitch alignment!)
KP_PITCH_ALIGN = 0.05 # Smaller Kp for pixel error
KI_PITCH_ALIGN = 0.001
KD_PITCH_ALIGN = 0.01

# KALMAN FILTER GAINS (tune these!)
Q_KF = 0.1
R_KF = 15 

# --- Camera Configuration (for RPi Cam v2 on Jetson / PC Webcam) ---
# Set desired resolution and FPS for your camera.
# For PC webcam, these values will be requested, but actual resolution might vary.
# For GStreamer, these MUST match the pipeline settings.
CAMERA_WIDTH = 640  # <--- Adjust this to your desired width (e.g., 640 for PC webcam, 1280 for RPi Cam)
CAMERA_HEIGHT = 480  # <--- Adjust this to your desired height (e.g., 480 for PC webcam, 720 for RPi Cam)
CAMERA_FPS = 30      # <--- Adjust this to your camera's FPS (e.g., 30)

# --- GStreamer Pipeline for RPi Cam v2 on Jetson (Uncomment to use) ---
# This pipeline is for CSI cameras like RPi Cam v2 on Jetson for optimal performance.
# Ensure 'nvarguscamsrc' is available and sensor_id is correct.
GSTREAMER_PIPELINE = (
    f"nvarguscamsrc sensor_id=0 ! "
    f"video/x-raw(memory:NVMM), width=(int){CAMERA_WIDTH}, height=(int){CAMERA_HEIGHT}, format=(string)NV12, framerate=(fraction){CAMERA_FPS}/1 ! "
    "nvvidconv flip-method=0 ! " # Adjust flip-method as needed (0=none, 1=horizontal, 2=vertical, 3=180-degree)
    f"video/x-raw, width=(int){CAMERA_WIDTH}, height=(int){CAMERA_HEIGHT}, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
)

# --- Camera Selection Flag ---
# Set to True to use GStreamer (RPi Cam v2 on Jetson).
# Set to False to use standard OpenCV webcam (PC webcam).
USE_GSTREAMER_CAMERA = False # <--- CHANGE THIS TO True OR False

# --- Video Recording Configuration ---
RECORD_VIDEO = True
VIDEO_FOLDER = "flights"
VIDEO_FPS = CAMERA_FPS # Match video recording FPS to camera FPS
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v') 

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
    """
    A simple 1D Kalman Filter for estimating a single state (e.g., horizontal position or vertical position).
    State: [position, velocity]
    Measurement: [raw_measurement]
    """
    def __init__(self, dt, Q, R, initial_pos=0.0):
        self.dt = dt
        self.Q = Q # Process noise covariance
        self.R = R # Measurement noise covariance

        # State vector [position, velocity]
        self.x = np.array([[initial_pos], [0.0]]) 
        
        # State transition matrix
        self.A = np.array([[1, dt],
                           [0, 1]])
        
        # Measurement matrix
        self.H = np.array([[1, 0]])
        
        # Process noise covariance matrix
        self.Q_matrix = np.array([[0.25*dt**4, 0.5*dt**3],
                                  [0.5*dt**3, dt**2]]) * self.Q
        
        # Measurement noise covariance matrix
        self.R_matrix = np.array([[self.R]])
        
        # Error covariance matrix (initialized with high uncertainty)
        self.P = np.array([[1000.0, 0.0],
                           [0.0, 1000.0]])

    def predict(self):
        """Predicts the next state."""
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q_matrix

    def update(self, z):
        """Updates the state with a new measurement."""
        y = z - np.dot(self.H, self.x) # Innovation (difference between measurement and prediction)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R_matrix # Innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman gain (how much to trust the measurement)
        
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_estimated_state(self):
        """Returns the estimated position and velocity."""
        return self.x[0, 0], self.x[1, 0] # Returns position, velocity
    
    def reset(self, initial_pos=0.0):
        """Resets the Kalman filter's state."""
        self.x = np.array([[initial_pos], [0.0]])
        self.P = np.array([[1000.0, 0.0], [0.0, 1000.0]]) # Reset uncertainty

# --- Placeholder for YOLOv8s Model and Mock Target Movement ---
# Global variables to simulate target movement for all classes
mock_target_pos = {
    0: {'x': 0, 'y': 0},
    1: {'x': 0, 'y': 0},
    2: {'x': 0, 'y': 0},
    3: {'x': 0, 'y': 0}
}

def initialize_mock_targets(width, height):
    """Initializes mock target positions based on image dimensions."""
    global mock_target_pos
    mock_target_pos[0] = {'x': width * 0.2, 'y': height * 0.3}
    mock_target_pos[1] = {'x': width * 0.5, 'y': height * 0.5}
    mock_target_pos[2] = {'x': width * 0.8, 'y': height * 0.7} # Priority 2
    mock_target_pos[3] = {'x': width * 0.3, 'y': height * 0.6}

def mock_yolo_inference(frame):
    """
    A placeholder function to simulate YOLOv8s object detection.
    This function generates multiple mock bounding boxes and class IDs
    to test the prioritization logic. It simulates some noise.
    """
    height, width, _ = frame.shape
    results = []

    # Simulate movement for all targets
    for class_id in mock_target_pos:
        mock_target_pos[class_id]['x'] += random.uniform(-5, 5) # Horizontal jitter
        mock_target_pos[class_id]['y'] += random.uniform(-3, 3) # Vertical jitter
        
        # Keep targets within bounds
        mock_target_pos[class_id]['x'] = np.clip(mock_target_pos[class_id]['x'], width * 0.1, width * 0.9)
        mock_target_pos[class_id]['y'] = np.clip(mock_target_pos[class_id]['y'], height * 0.1, height * 0.9)

    # Generate bounding boxes and add to results (with random chance of appearance)
    def create_mock_detection(x_center_m, y_center_m, class_id):
        # Add some random noise to the detection itself
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
    # The higher the probability, the more likely this class will be detected
    # Adjust these probabilities to test prioritization
    if random.random() < 0.9: # High chance for class 2 (highest priority)
        results.append(create_mock_detection(mock_target_pos[2]['x'], mock_target_pos[2]['y'], 2))
    
    if random.random() < 0.7: # Medium chance for class 1
        results.append(create_mock_detection(mock_target_pos[1]['x'], mock_target_pos[1]['y'], 1))

    if random.random() < 0.6: # Lower chance for class 3
        results.append(create_mock_detection(mock_target_pos[3]['x'], mock_target_pos[3]['y'], 3))
    
    if random.random() < 0.5: # Even lower chance for class 0
        results.append(create_mock_detection(mock_target_pos[0]['x'], mock_target_pos[0]['y'], 0))

    return results

# --- Drone Control Functions (MAVSDK) ---
async def arm_and_takeoff(drone, aTargetAltitude):
    """Arms vehicle and flies to aTargetAltitude using MAVSDK offboard velocity commands."""
    print("Waiting for drone to connect...")
    async for state in drone.telemetry.health():
        if state.is_gyrometer_calibration_ok and state.is_accelerometer_calibration_ok and state.is_magnetometer_calibration_ok and state.is_home_position_ok:
            print("Drone is healthy and ready to arm.")
            break
        await asyncio.sleep(1)

    # Added robust checks for landed state and disarmed state before arming
 

    # --- Add a small delay and set to HOLD mode before arming ---
    print("Giving SITL a moment to stabilize and setting to HOLD mode...")
    await asyncio.sleep(2) # Give SITL a couple of seconds
    try:
        await drone.action.hold() # Set to HOLD mode
        print("Drone set to HOLD mode.")
        await asyncio.sleep(1) # Give drone time to switch mode
    except Exception as e:
        print(f"Failed to set drone to HOLD mode: {e}. Attempting arm anyway.")
    # --- End added delay and HOLD mode ---

    print("Arming motors...")
    try:
        await drone.action.arm()
        print("Motors armed successfully.")
        await asyncio.sleep(1) # Give drone time to arm
    except Exception as e: # Catching generic Exception to see all arming errors
        print(f"Arming failed with error: {e}. Check SITL console for pre-arm messages.")
        return False # Indicate failure to the caller if arming fails

    # --- Takeoff using direct thrust command in offboard mode ---
    # Send a neutral attitude rate setpoint with takeoff thrust before starting offboard
    await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, TAKEOFF_THRUST_VALUE))
    try:
        await drone.offboard.start()
        print(f"Offboard mode started for takeoff with thrust: {TAKEOFF_THRUST_VALUE}")
        # Debug: Confirm current flight mode after offboard start
        current_mode_after_offboard = None
        async for fm in drone.telemetry.flight_mode():
            current_mode_after_offboard = fm
            break
        print(f"Current flight mode after offboard.start(): {current_mode_after_offboard.name}")

    except OffboardError as error:
        print(f"Starting offboard mode failed for takeoff with error: {error}. Please check drone state and mode.")
        return False # Indicate failure to the caller

    print(f"Taking off to {aTargetAltitude} meters by commanding thrust!")
    takeoff_start_time = time.time()
    TAKE_OFF_TIMEOUT = 15 # seconds to reach altitude

    # Loop to maintain takeoff thrust until altitude is reached
    while True:
        current_altitude = 0.0
        async for position in drone.telemetry.position():
            current_altitude = position.relative_altitude_m
            break # Get current altitude and break from this inner loop

        print(f"Altitude: {current_altitude:.2f}m / Target: {aTargetAltitude:.2f}m")
        # Keep sending takeoff thrust until altitude is reached
        await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, TAKEOFF_THRUST_VALUE)) 
        
        if current_altitude >= aTargetAltitude * 0.95:
            print(f"Reached target altitude: {current_altitude:.2f}m")
            break
        if (time.time() - takeoff_start_time) > TAKE_OFF_TIMEOUT:
            print(f"Takeoff timed out after {TAKE_OFF_TIMEOUT} seconds. Current altitude: {current_altitude:.2f}m")
            return False # Indicate takeoff failure
        await asyncio.sleep(0.1) # Shorter sleep for smoother altitude check
    # Check if altitude cap is reached
        if current_altitude >= ALTITUDE_CAP_M:
            print(f"Altitude cap of {ALTITUDE_CAP_M:.2f}m reached. Stopping upward movement.")
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0)) # Stop vertical movement
            await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, HOVER_THRUST_VALUE)) # Switch to hover thrust
            break 

    # After reaching altitude, set thrust to hover value
    await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, HOVER_THRUST_VALUE))
    print("Hovering at target altitude.")
    return True # Indicate success


async def send_attitude_rate_commands(drone, roll_rate, pitch_rate, yaw_rate, thrust_value):
    """Sends roll, pitch, and yaw rate commands to the drone using MAVSDK offboard."""
    # Clip rates to their configured limits
    roll_rate = np.clip(roll_rate, -PITCH_LIMIT_ALIGN, PITCH_LIMIT_ALIGN) # Using Pitch_Limit for roll as well
    pitch_rate = np.clip(pitch_rate, -PITCH_LIMIT_ALIGN, PITCH_LIMIT_ALIGN)
    yaw_rate = np.clip(yaw_rate, -YAW_LIMIT, YAW_LIMIT)

    # MAVSDK's AttitudeRate expects radians per second
    # For vertical alignment: If target is below center (positive error_align), need to pitch nose DOWN (camera looks down).
    # MAVSDK's positive pitch rate = nose up. So if target is below (positive error_align), we want nose down (negative pitch_rate_rad_s).
    # Therefore, invert the pitch_rate_cmd.
    
    attitude_rate = AttitudeRate(
        np.deg2rad(roll_rate),
        np.deg2rad(-pitch_rate), # Invert pitch_rate here to make positive error_align -> nose down
        np.deg2rad(yaw_rate),
        thrust_value # Thrust from 0.0 to 1.0
    )
    await drone.offboard.set_attitude_rate(attitude_rate)

# --- Main Tracking Loop ---
async def tracking_loop():
    print("Starting tracking loop...")
    drone = System()
    await drone.connect(system_address=CONNECTION_STRING)

    # Give the drone a bit of time to connect and get telemetry
    print("Waiting for drone telemetry...")
    async for state in drone.telemetry.health():
        if state.is_gyrometer_calibration_ok and state.is_accelerometer_calibration_ok and state.is_magnetometer_calibration_ok and state.is_home_position_ok:
            print("Drone health checks passed.")
            break
        await asyncio.sleep(1)

    print("Waiting for drone to be ready for offboard (switch to GUIDED mode in Mission Planner)...")
    while True:
        async for flight_mode_telemetry in drone.telemetry.flight_mode():
            # Corrected: Use FlightMode.OFFBOARD for companion control in MAVSDK
            if flight_mode_telemetry == FlightMode.OFFBOARD:
                print("Drone is in OFFBOARD mode (equivalent to GUIDED for companion control). Preparing for offboard control.")
                break
            print(f"Current mode: {flight_mode_telemetry.name}. Waiting for OFFBOARD (or switch to GUIDED in Mission Planner).")
            await asyncio.sleep(1)
        # Ensure the loop breaks if mode is OFFBOARD
        if flight_mode_telemetry == FlightMode.OFFBOARD:
            break

    # Modified takeoff sequence
    takeoff_successful = await arm_and_takeoff(drone, 2.0) # Take off to 2 meters
    if not takeoff_successful:
        print("Takeoff failed. Exiting tracking loop.")
        return # Exit if takeoff wasn't successful

    yaw_pid_controller = PID(KP_YAW, KI_YAW, KD_YAW)
    # Pitch PID for vertical pixel alignment
    pitch_align_pid_controller = PID(KP_PITCH_ALIGN, KI_PITCH_ALIGN, KD_PITCH_ALIGN)
    
    # --- Camera Initialization ---
    if USE_GSTREAMER_CAMERA:
        cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
        print(f"Attempting to open GStreamer camera: {GSTREAMER_PIPELINE}")
    else:
        cap = cv2.VideoCapture(0) # Use 0 for default webcam on PC
        print(f"Attempting to open default webcam (index 0).")

    if not cap.isOpened():
        print("Error: Could not open camera.")
        print("Please check:")
        if USE_GSTREAMER_CAMERA:
            print("  1. Is the RPi Cam v2 connected and enabled on Jetson?")
            print("  2. Are GStreamer plugins installed (e.g., 'sudo apt install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav')?")
            print("  3. Are your user permissions correct for camera access (e.g., 'sudo usermod -a -G video $USER')?")
            print("  4. Is the 'nvarguscamsrc' plugin working (test with 'gst-launch-1.0 nvarguscamsrc ! nvvidconv ! xvimagesink')?")
        else:
            print("  1. Is your PC webcam connected and not in use by another application?")
            print("  2. Try a different camera index if you have multiple webcams (e.g., cv2.VideoCapture(1)).")
        return # Exit if camera fails

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read initial frame from camera. Exiting loop.")
        try: await drone.offboard.stop()
        except OffboardError as error: print(f"Failed to stop offboard: {error}")
        return

    image_height, image_width, _ = frame.shape
    # Ensure image_width and image_height match configured resolution
    if image_width != CAMERA_WIDTH or image_height != CAMERA_HEIGHT:
        print(f"Warning: Camera resolution ({image_width}x{image_height}) does not match configured CAMERA_WIDTH/HEIGHT ({CAMERA_WIDTH}x{CAMERA_HEIGHT}). Forcing consistency.")
        # For consistency, force these to match if there's a discrepancy
        image_width = CAMERA_WIDTH
        image_height = CAMERA_HEIGHT

    image_center_x = image_width / 2
    image_center_y = image_height / 2 

    # Initialize mock target positions once with image dimensions
    initialize_mock_targets(image_width, image_height)

    # --- Real YOLO Model Loading ---
    yolo_session = None # Use yolo_session for ONNX Runtime
    onnx_input_height, onnx_input_width = 0, 0 # Initialize for scope
    yolo_input_name = ""
    yolo_output_names = []

    _use_real_yolo_local = USE_REAL_YOLO 
    
    if _use_real_yolo_local:
        try:
            # Configure ONNX Runtime to use TensorRT or CUDA execution providers
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            sess_options = ort.SessionOptions()
            
            yolo_session = ort.InferenceSession(REAL_YOLO_MODEL_PATH, sess_options, providers=providers)
            
            # Get input/output names and expected input size
            yolo_input_name = yolo_session.get_inputs()[0].name
            input_shape = yolo_session.get_inputs()[0].shape
            # Assuming NCHW format (Batch, Channels, Height, Width)
            onnx_input_height, onnx_input_width = input_shape[2], input_shape[3] 
            yolo_output_names = [output.name for output in yolo_session.get_outputs()]
            
            print(f"TensorRT/ONNX model loaded from: {REAL_YOLO_MODEL_PATH}")
            print(f"YOLO input name: {yolo_input_name}, input shape: {input_shape}, output names: {yolo_output_names}")

        except ImportError:
            print("Error: 'onnxruntime' not found. Please install it with 'pip install onnxruntime-gpu'. Falling back to mock YOLO.")
            _use_real_yolo_local = False
        except Exception as e:
            print(f"Error loading ONNX/TensorRT model: {e}. Falling back to mock YOLO.")
            _use_real_yolo_local = False
    # --- End Real YOLO Model Loading ---

    # Initialize Kalman Filters for yaw (horizontal position) and pitch (vertical position)
    yaw_kalman_filter = KalmanFilter(dt=0.1, Q=Q_KF, R=R_KF, initial_pos=image_center_x) 
    pitch_kalman_filter = KalmanFilter(dt=0.1, Q=Q_KF, R=R_KF, initial_pos=image_center_y) # For vertical alignment
    
    current_time = time.time()
    last_time = current_time

    # --- Data Logging Setup ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    flight_folder = os.path.join(VIDEO_FOLDER, f"flight_data_{timestamp}")
    os.makedirs(flight_folder, exist_ok=True)
    data_filename = os.path.join(flight_folder, f"flight_log_{timestamp}.csv")

    csv_headers = [
        'Time_s', 'FlightMode', 
        'Raw_Yaw_Err_px', 'KF_Yaw_Est_px', 'Yaw_Cmd_deg_s', 'Yaw_I_term',
        'Raw_Pitch_Align_Err_px', 'KF_Pitch_Align_Est_px', 'Pitch_Cmd_deg_s', 'Pitch_I_term', # Changed for vertical alignment
        'Target_Class_ID',
        'Current_Altitude_m' # Added altitude to log
    ]
    csv_file = open(data_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_headers)
    # --- End Data Logging Setup ---

    # --- Video Recording Setup ---
    video_writer = None
    if RECORD_VIDEO:
        video_filename = os.path.join(flight_folder, f"tracking_video_{timestamp}.mp4")
        video_writer = cv2.VideoWriter(video_filename, VIDEO_CODEC, VIDEO_FPS, (image_width, image_height))
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {video_filename}. Recording disabled.")
            video_writer = None
    # --- End Video Recording Setup ---

    is_offboard_active_from_script = True # Assume offboard is active after start()

    while True:
        # Debug print
        # print(f"Main loop running. Time: {current_time:.2f}") # Moved after current_time update
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera. Exiting loop.")
            break

        current_time = time.time() # Update current_time for the next iteration
        dt = current_time - last_time
        last_time = current_time
        print(f"Main loop running. Time: {current_time:.2f}, dt: {dt:.4f}") # Debug print

        yaw_kalman_filter.dt = dt
        yaw_kalman_filter.A = np.array([[1, dt], [0, 1]])
        yaw_kalman_filter.Q_matrix = np.array([[0.25*dt**4, 0.5*dt**3], [0.5*dt**3, dt**2]]) * yaw_kalman_filter.Q

        pitch_kalman_filter.dt = dt
        pitch_kalman_filter.A = np.array([[1, dt], [0, 1]])
        pitch_kalman_filter.Q_matrix = np.array([[0.25*dt**4, 0.5*dt**3], [0.5*dt**3, dt**2]]) * pitch_kalman_filter.Q

        yaw_kalman_filter.predict()
        pitch_kalman_filter.predict()

        yaw_rate_cmd = 0.0
        pitch_rate_cmd = 0.0
        
        current_flight_mode = None
        async for flight_mode_telemetry in drone.telemetry.flight_mode():
            current_flight_mode = flight_mode_telemetry
            break

        # Get current altitude for logging and display
        current_altitude_m = 0.0
        async for position_telemetry in drone.telemetry.position():
            current_altitude_m = position_telemetry.relative_altitude_m
            break

        log_data_row = [current_time] # Start logging data for this frame
        log_data_row.append(current_flight_mode.name) # Add flight mode

        # Corrected: Check for FlightMode.OFFBOARD
        if current_flight_mode == FlightMode.OFFBOARD:
            # Offboard mode is already started by arm_and_takeoff.
            # This check is primarily to ensure it remains active if it somehow gets stopped.
            if not is_offboard_active_from_script:
                try:
                    await drone.offboard.start()
                    print("Offboard mode restarted due to OFFBOARD mode detection.")
                    is_offboard_active_from_script = True
                except OffboardError as error:
                    print(f"Failed to restart offboard: {error}. Staying in safe state.")
                    await send_attitude_rate_commands(drone, 0, 0, 0, HOVER_THRUST_VALUE) # Use hover thrust if offboard fails to restart
                    # Log default values if offboard failed to start
                    log_data_row.extend([0, 0, 0, 0, 0, 0, 0, 0, -1, current_altitude_m]) 
                    csv_writer.writerow(log_data_row)
                    if video_writer is not None: video_writer.write(frame)
                    cv2.imshow("Object Tracking", frame)
                    await asyncio.sleep(0.01)
                    continue

            # --- Perform YOLO Inference (Real or Mock) ---
            detections = []
            if _use_real_yolo_local and yolo_session: # Use the local flag here
                try:
                    # Preprocess frame for YOLO
                    img = cv2.resize(frame, (onnx_input_width, onnx_input_height))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    img = np.transpose(img, (2, 0, 1))
                    img = np.expand_dims(img, axis=0) # Add batch dimension

                    # Run inference
                    outputs = yolo_session.run(yolo_output_names, {yolo_input_name: img})
                    
                    # --- POST-PROCESSING FOR YOLOv8/v11 ONNX OUTPUT ---
                    # This is CRITICAL and model-specific. The raw 'outputs' tensor needs to be converted
                    # into a list of bounding boxes, confidence scores, and class IDs.
                    # Ultralytics YOLOv8 ONNX output typically has a shape like (1, num_detections, 6)
                    # where 6 is [x_center, y_center, width, height, confidence, class_id] (or similar)
                    # You will need to implement Non-Max Suppression (NMS) and confidence filtering.
                    
                    # Placeholder for actual post-processing logic
                    # This is a highly simplified example; you'll need a full implementation.
                    raw_output = outputs[0] # Assuming first output tensor contains detections
                    
                    # Example: Filter by confidence and convert to xyxy format
                    # You will need to adapt this based on your exact model output structure
                    for det in raw_output[0]: # Assuming batch size 1
                        confidence = det[4] # Example: confidence is 5th element
                        class_id = det[5] # Example: class ID is 6th element (for single class output)
                        # For multi-class output, it might be np.argmax(det[5:])
                        
                        if confidence > 0.5: # Confidence threshold
                            # Convert (x_center, y_center, width, height) to (x1, y1, x2, y2)
                            # And scale coordinates back to original frame size
                            
                            x_center_norm, y_center_norm, w_norm, h_norm = det[0], det[1], det[2], det[3]
                            
                            # Scale coordinates from model input size to original frame size
                            x1 = int((x_center_norm - w_norm/2) / onnx_input_width * image_width)
                            y1 = int((y_center_norm - h_norm/2) / onnx_input_height * image_height)
                            x2 = int((x_center_norm + w_norm/2) / onnx_input_width * image_width)
                            y2 = int((y_center_norm + h_norm/2) / onnx_input_height * image_height)

                            detections.append({
                                'xyxy': [x1, y1, x2, y2],
                                'cls': int(class_id)
                            })
                    # You might also need to apply NMS here if your ONNX model doesn't include it.

                except Exception as e:
                    print(f"Error during real YOLO inference with ONNX Runtime: {e}. Using mock detections for this frame.")
                    detections = mock_yolo_inference(frame)
            else:
                detections = mock_yolo_inference(frame)
            # --- End YOLO Inference ---
            
            selected_box = None
            highest_priority_score = -1

            for detection in detections:
                class_id = detection['cls']
                priority_score = CLASS_PRIORITIES.get(class_id, -1) 
                
                if priority_score > highest_priority_score:
                    highest_priority_score = priority_score
                    selected_box = detection
            
            # --- Initialize logging variables for this loop iteration ---
            raw_yaw_err = 0.0
            kf_yaw_est = yaw_kalman_filter.get_estimated_state()[0]
            yaw_cmd = 0.0
            yaw_i_term = yaw_pid_controller.I_term

            raw_pitch_align_err = 0.0
            kf_pitch_align_est = pitch_kalman_filter.get_estimated_state()[0]
            pitch_cmd = 0.0
            pitch_i_term = pitch_align_pid_controller.I_term
            
            tracked_class_id = -1

            if selected_box is not None:
                x1, y1, x2, y2 = selected_box['xyxy']
                raw_x_center_bbox = (x1 + x2) / 2
                raw_y_center_bbox = (y1 + y2) / 2 
                
                yaw_kalman_filter.update(np.array([[raw_x_center_bbox]]))
                pitch_kalman_filter.update(np.array([[raw_y_center_bbox]]))
                
                estimated_x_center, _ = yaw_kalman_filter.get_estimated_state()
                estimated_y_center, _ = pitch_kalman_filter.get_estimated_state()

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

                await send_attitude_rate_commands(drone, 0, pitch_rate_cmd, yaw_rate_cmd, HOVER_THRUST_VALUE) # Use HOVER_THRUST_VALUE for tracking

                # --- Update logging variables with current values ---
                raw_yaw_err = raw_x_center_bbox - image_center_x
                kf_yaw_est = estimated_x_center
                yaw_cmd = yaw_rate_cmd
                yaw_i_term = yaw_pid_controller.I_term

                raw_pitch_align_err = raw_y_center_bbox - image_center_y
                kf_pitch_align_est = estimated_y_center
                pitch_cmd = pitch_rate_cmd
                pitch_i_term = pitch_align_pid_controller.I_term
                tracked_class_id = selected_box['cls']

                # --- Visualization ---
                for detection in detections:
                    dx1, dy1, dx2, dy2 = detection['xyxy']
                    d_class_id = detection['cls']
                    color = (0, 100, 255) if detection != selected_box else (0, 255, 0)
                    cv2.rectangle(frame, (int(dx1), int(dy1)), (int(dx2), int(dy2)), color, 2)
                    cv2.putText(frame, f"ID:{d_class_id}", (int(dx1), int(dy1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (int(raw_x_center_bbox), int(raw_y_center_bbox)), 5, (0, 0, 255), -1) # Raw tracked center (red)
                cv2.circle(frame, (int(estimated_x_center), int(estimated_y_center)), 8, (255, 0, 255), -1) # Estimated center (magenta)
                cv2.line(frame, (int(image_center_x), 0), (int(image_center_x), image_height), (255, 0, 0), 2) # Image center vertical (red)
                cv2.line(frame, (0, int(image_center_y)), (image_width, int(image_center_y)), (255, 0, 0), 2) # Image center horizontal (red)


                cv2.putText(frame, f"Tracking Class ID: {tracked_class_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Yaw Err: {yaw_error:.2f}px | Cmd: {yaw_rate_cmd:.1f}deg/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Pitch Align Err: {pitch_error_align:.2f}px | Cmd: {pitch_rate_cmd:.1f}deg/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Altitude: {current_altitude_m:.2f}m / Target: {2.0:.2f}m", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2) # Added altitude display
            else:
                # If no target found, but in companion-control mode, use Kalman prediction for motion
                estimated_x_center, _ = yaw_kalman_filter.get_estimated_state()
                estimated_y_center, _ = pitch_kalman_filter.get_estimated_state()
                
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
                
                await send_attitude_rate_commands(drone, 0, pitch_rate_cmd, yaw_rate_cmd, HOVER_THRUST_VALUE) # Use HOVER_THRUST_VALUE for predicting

                # --- Update logging variables with current values (no target) ---
                raw_yaw_err = 0.0
                kf_yaw_est = estimated_x_center
                yaw_cmd = yaw_rate_cmd
                yaw_i_term = yaw_pid_controller.I_term

                raw_pitch_align_err = 0.0
                kf_pitch_align_est = estimated_y_center
                pitch_cmd = 0.0
                pitch_i_term = pitch_align_pid_controller.I_term
                tracked_class_id = -1


                cv2.circle(frame, (int(estimated_x_center), int(estimated_y_center)), 8, (255, 165, 0), -1) # Estimated center (orange)
                cv2.line(frame, (int(image_center_x), 0), (int(image_center_x), image_height), (255, 0, 0), 2) # Image center vertical (red)
                cv2.line(frame, (0, int(image_center_y)), (image_width, int(image_center_y)), (255, 0, 0), 2) # Image center horizontal (red)
                cv2.putText(frame, "No Target Detected - Predicting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Yaw Err: {yaw_error:.2f}px | Cmd: {yaw_rate_cmd:.1f}deg/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Pitch Align Err: {pitch_error_align:.2f}px | Cmd: {pitch_rate_cmd:.1f}deg/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Altitude: {current_altitude_m:.2f}m / Target: {2.0:.2f}m", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2) # Added altitude display
            
            cv2.putText(frame, "Tracking: ON", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        else: # Not in OFFBOARD mode
            if is_offboard_active_from_script:
                try:
                    await drone.offboard.stop()
                    print("Offboard mode stopped due to mode change.")
                    is_offboard_active_from_script = False
                except OffboardError as error:
                    print(f"Failed to stop offboard: {error}")
            
            await send_attitude_rate_commands(drone, 0, 0, 0, HOVER_THRUST_VALUE) # Send zero rates with hover thrust if offboard not active, as a safety
            yaw_pid_controller.reset()
            pitch_align_pid_controller.reset()
            yaw_kalman_filter.reset(initial_pos=image_center_x)
            pitch_kalman_filter.reset(initial_pos=image_center_y)

            # --- Update logging variables when tracking is OFF ---
            raw_yaw_err = 0.0
            kf_yaw_est = image_center_x
            yaw_cmd = 0.0
            yaw_i_term = 0.0

            raw_pitch_align_err = 0.0
            kf_pitch_align_est = image_center_y
            pitch_cmd = 0.0
            pitch_i_term = 0.0
            tracked_class_id = -1

            cv2.putText(frame, f"Tracking: OFF ({current_flight_mode.name} mode)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # --- Add data to the log row and write to CSV ---
        log_data_row.extend([
            raw_yaw_err, kf_yaw_est, yaw_cmd, yaw_i_term,
            raw_pitch_align_err, kf_pitch_align_est, pitch_cmd, pitch_i_term,
            tracked_class_id,
            current_altitude_m # Added altitude to log
        ])
        csv_writer.writerow(log_data_row)
        # --- End Data Logging ---

        if video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Stopping and landing the drone.")
    if is_offboard_active_from_script:
        try:
            await drone.offboard.stop()
            print("Offboard mode stopped before landing.")
        except OffboardError as error:
            print(f"Failed to stop offboard before landing: {error}")

    await drone.action.land()
    await asyncio.sleep(5)
    await drone.action.disarm()
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {video_filename}")
    csv_file.close() # Close the CSV file
    print(f"Flight log saved to: {data_filename}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(tracking_loop())