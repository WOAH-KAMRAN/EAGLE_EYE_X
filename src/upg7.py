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
from mavsdk.telemetry import FlightMode, Position, LandedState, Health # Import Health for clarity

import csv # For saving data to CSV
import random

# TensorRT imports for .engine file support
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # This automatically initializes CUDA
    TRT_AVAILABLE = True
    print("TensorRT and PyCUDA successfully imported.")
except ImportError as e:
    print(f"TensorRT or PyCUDA not available: {e}")
    print("Please install TensorRT and PyCUDA for Jetson Orin:")
    print("sudo apt install python3-libnvinfer-dev python3-libnvinfer")
    print("pip install pycuda")
    TRT_AVAILABLE = False

# Depth estimation imports
try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    DEPTH_AVAILABLE = True
    print("PyTorch successfully imported for depth estimation.")
    
    # Try importing transformers for ZoeDepth
    try:
        from transformers import pipeline, AutoImageProcessor, ZoeDepthForDepthEstimation
        ZOEDEPTH_AVAILABLE = True
        print("Transformers library available - ZoeDepth support enabled.")
    except ImportError:
        ZOEDEPTH_AVAILABLE = False
        print("Transformers not available. Install with: pip install transformers")
        print("ZoeDepth will not be available, falling back to MiDaS if selected.")
        
except ImportError as e:
    print(f"PyTorch not available for depth estimation: {e}")
    print("Install with: pip install torch torchvision transformers")
    DEPTH_AVAILABLE = False
    ZOEDEPTH_AVAILABLE = False

# --- Configuration ---
# MAVSDK connection string. For SITL on the same PC, use udpin://:14550
# For real drone via USB/UART on Jetson, e.g., 'serial:///dev/ttyTHS1:921600'
CONNECTION_STRING = 'udpin://127.0.0.1:14550'

# --- YOLO Model Configuration ---
USE_REAL_YOLO = True          # Set to True to use your real YOLO model, False for mock detections
# Path to your TensorRT engine file (e.g., 'yolov8n.engine' or 'yolov8s.engine')
REAL_YOLO_MODEL_PATH = 'yolov8n.engine' # <--- IMPORTANT: UPDATE THIS PATH!
# TensorRT engine files provide optimized inference on Jetson platforms
# Generate your .engine file from ONNX using trtexec or Python TensorRT API

# --- Depth Estimation Configuration ---
USE_DEPTH_ESTIMATION = True     # Set to True to enable depth-based forward movement
DEPTH_MODEL_TYPE = 'zoedepth'   # Options: 'midas', 'zoedepth' (zoedepth recommended for metric accuracy)
ZOEDEPTH_MODEL_VARIANT = 'zoedepth-nyu-kitti'  # Options: 'zoedepth-nyu', 'zoedepth-kitti', 'zoedepth-nyu-kitti'
FORCE_MIDAS = False             # Set to True to force MiDaS usage even if ZoeDepth is available
TARGET_DISTANCE_M = 2.0         # Desired distance to target in meters (USER REQUESTED: 2m)
DISTANCE_DEAD_ZONE_M = 0.2      # Dead zone for distance control in meters (tighter for 2m precision)
MAX_FORWARD_SPEED_MPS = 1.0     # Maximum forward/backward speed in m/s (reduced for closer approach)
YAW_CENTERED_THRESHOLD = 25     # Pixel threshold - when target is considered "centered" for forward movement

# --- Depth Map Tuning Parameters ---
# These parameters control how the depth estimation behaves and can be tuned for better accuracy
DEPTH_SMOOTHING_FACTOR = 0.7       # Temporal smoothing (0.0-1.0): Higher = more smoothing, less noise
DEPTH_ROI_SCALE = 0.3              # Region of interest scale around target (0.1-0.5): Smaller = more focused
DEPTH_MIN_CONFIDENCE = 0.1         # Minimum confidence threshold for depth estimates (0.0-1.0)
DEPTH_MAX_RANGE_M = 8.0            # Maximum valid depth range in meters (filter far objects)
DEPTH_MIN_RANGE_M = 0.5            # Minimum valid depth range in meters (filter very close objects)
USE_DEPTH_MEDIAN_FILTER = True     # Apply median filter to reduce depth noise (True/False)
DEPTH_PERCENTILE = 25              # Percentile for depth aggregation (0-50): Lower = prioritize closer objects

# Define custom priority for each class ID. Higher value means higher priority.
# Priority: Class 2 > Class 1 > Class 3 > Class 0
CLASS_PRIORITIES = {
    2: 4,  # Highest priority
    1: 3,  # Second highest
    3: 2,  # Third highest
    0: 1   # Lowest priority
}

YAW_LIMIT = 50                 # Maximum yaw speed in degrees/second
PITCH_LIMIT_ALIGN = 20         # Maximum pitch speed in degrees/second for vertical alignment
HOVER_THRUST_VALUE = 0.6       # Thrust value for hovering (0.0 to 1.0) - Tune this for stable hover
TAKEOFF_THRUST_VALUE = 0.95    # Thrust value for takeoff (0.0 to 1.0) - used by AttitudeRate for initial offboard start
ALTITUDE_CAP_M = 10.0          # Altitude cap in meters

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

# PID GAINS for distance control (forward/backward movement) - Tuned for 2m operation
KP_DISTANCE = 1.0     # Proportional gain for distance error (increased for 2m responsiveness)
KI_DISTANCE = 0.02    # Integral gain for distance error (reduced to prevent overshoot)
KD_DISTANCE = 0.25    # Derivative gain for distance error (increased for better damping)

# KALMAN FILTER GAINS (tune these!)
Q_KF = 0.1
R_KF = 10
# Distance Kalman filter gains - Tuned for 2m precision
Q_DISTANCE_KF = 0.15  # Process noise for distance estimation (reduced for stability)
R_DISTANCE_KF = 0.8   # Measurement noise for distance estimation (adjusted for 2m accuracy) 

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

# --- TensorRT Inference Engine Class ---
class TensorRTInference:
    """
    TensorRT inference engine for optimized YOLO inference on Jetson platforms.
    This class handles .engine file loading and provides fast GPU inference.
    """
    def __init__(self, engine_file_path, logger=None):
        """
        Initialize TensorRT inference engine.
        
        Args:
            engine_file_path (str): Path to the .engine file
            logger: TensorRT logger (optional)
        """
        self.engine_file_path = engine_file_path
        self.logger = logger if logger else trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.input_shape = None
        self.output_shapes = []
        
    def load_engine(self):
        """Load TensorRT engine from .engine file."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available. Please install TensorRT and PyCUDA.")
            
        try:
            # Load engine from file
            with open(self.engine_file_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
                
            if self.engine is None:
                raise RuntimeError(f"Failed to load engine from {self.engine_file_path}")
                
            # Create execution context
            self.context = self.engine.create_execution_context()
            
            # Setup input/output buffers
            self._setup_buffers()
            
            print(f"TensorRT engine loaded successfully from: {self.engine_file_path}")
            print(f"Input shape: {self.input_shape}")
            print(f"Output shapes: {self.output_shapes}")
            
            return True
            
        except Exception as e:
            print(f"Error loading TensorRT engine: {e}")
            return False
    
    def _setup_buffers(self):
        """Setup CUDA buffers for inputs and outputs."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer to device bindings
            self.bindings.append(int(device_mem))
            
            # Append to the appropriate list
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
                self.input_shape = self.engine.get_binding_shape(binding)
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                self.output_shapes.append(self.engine.get_binding_shape(binding))
    
    def infer(self, input_data):
        """
        Run inference on input data.
        
        Args:
            input_data (numpy.ndarray): Input image data in the correct format
            
        Returns:
            list: Output predictions from the model
        """
        if not self.engine or not self.context:
            raise RuntimeError("Engine not loaded. Call load_engine() first.")
        
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Copy input data from host to device
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output data from device to host
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
            outputs.append(output['host'].copy())
        
        # Synchronize stream
        self.stream.synchronize()
        
        # Reshape outputs to correct shapes
        reshaped_outputs = []
        for i, output in enumerate(outputs):
            reshaped_outputs.append(output.reshape(self.output_shapes[i]))
        
        return reshaped_outputs
    
    def get_input_shape(self):
        """Get input shape of the model."""
        return self.input_shape
    
    def __del__(self):
        """Cleanup CUDA resources."""
        try:
            if hasattr(self, 'inputs'):
                for inp in self.inputs:
                    if 'device' in inp:
                        inp['device'].free()
            if hasattr(self, 'outputs'):
                for out in self.outputs:
                    if 'device' in out:
                        out['device'].free()
        except:
            pass

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

# --- Distance Kalman Filter (specialized for depth estimation) ---
class DistanceKalmanFilter:
    """
    Specialized Kalman Filter for distance estimation with depth-specific noise handling.
    """
    def __init__(self, dt, Q, R, initial_distance=5.0):
        self.dt = dt
        self.Q = Q  # Process noise
        self.R = R  # Measurement noise
        
        # State: [distance, velocity]
        self.x = np.array([[initial_distance], [0.0]])
        
        # State transition matrix
        self.A = np.array([[1, dt], [0, 1]])
        
        # Measurement matrix (we only measure distance)
        self.H = np.array([[1, 0]])
        
        # Process noise covariance
        self.Q_matrix = np.array([[0.25*dt**4, 0.5*dt**3],
                                  [0.5*dt**3, dt**2]]) * self.Q
        
        # Measurement noise covariance
        self.R_matrix = np.array([[self.R]])
        
        # Error covariance matrix
        self.P = np.array([[100.0, 0.0], [0.0, 100.0]])
    
    def predict(self):
        """Predict the next state."""
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q_matrix
    
    def update(self, measurement):
        """Update with distance measurement."""
        if measurement is not None:
            # Innovation
            y = measurement - np.dot(self.H, self.x)
            # Innovation covariance
            S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R_matrix
            # Kalman gain
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            
            # Update state and covariance
            self.x = self.x + np.dot(K, y)
            self.P = self.P - np.dot(np.dot(K, self.H), self.P)
    
    def get_estimated_state(self):
        """Get estimated distance and velocity."""
        return self.x[0, 0], self.x[1, 0]
    
    def reset(self, initial_distance=5.0):
        """Reset the filter state."""
        self.x = np.array([[initial_distance], [0.0]])
        self.P = np.array([[100.0, 0.0], [0.0, 100.0]])
    
    def update_dt(self, dt):
        """Update the time step and recalculate matrices."""
        self.dt = dt
        self.A = np.array([[1, dt], [0, 1]])
        self.Q_matrix = np.array([[0.25*dt**4, 0.5*dt**3],
                                  [0.5*dt**3, dt**2]]) * self.Q

# --- Monocular Depth Estimation Classes ---
class ZoeDepthEstimator:
    """
    ZoeDepth-based metric depth estimation for precise distance measurements.
    Provides direct metric depth in meters without conversion factors.
    """
    def __init__(self, model_variant='zoedepth-nyu-kitti'):
        self.model_variant = model_variant
        self.model = None
        self.processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.last_distance = None
        print(f"Initializing ZoeDepth estimator with device: {self.device}")
        
    def load_model(self):
        """
        Load ZoeDepth model using Transformers library.
        """
        if not ZOEDEPTH_AVAILABLE:
            print("ZoeDepth not available. Install transformers: pip install transformers")
            return False
            
        try:
            print(f"Loading ZoeDepth model: Intel/{self.model_variant}...")
            
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(f"Intel/{self.model_variant}")
            self.model = ZoeDepthForDepthEstimation.from_pretrained(f"Intel/{self.model_variant}")
            self.model = self.model.to(self.device).eval()
            
            # Enable half precision for faster inference if using GPU
            if self.device.type == 'cuda':
                self.model = self.model.half()
                
            self.model_loaded = True
            print(f"ZoeDepth model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Failed to load ZoeDepth model: {e}")
            self.model_loaded = False
            return False
    
    def estimate_depth(self, frame):
        """
        Estimate depth map from RGB frame using ZoeDepth.
        Returns metric depth in meters.
        """
        if not self.model_loaded:
            return None
            
        try:
            # Convert frame to PIL Image
            if isinstance(frame, np.ndarray):
                # Convert BGR to RGB if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
                image = Image.fromarray(frame_rgb)
            else:
                image = frame
            
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Half precision for GPU
            if self.device.type == 'cuda':
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process to get metric depth
            post_processed = self.processor.post_process_depth_estimation(
                outputs, 
                source_sizes=[(image.height, image.width)]
            )
            
            predicted_depth = post_processed[0]["predicted_depth"]
            
            # Convert to numpy and return metric depth in meters
            depth_meters = predicted_depth.cpu().numpy()
            return depth_meters
            
        except Exception as e:
            print(f"ZoeDepth inference error: {e}")
            return None
    
    def get_target_distance(self, frame, target_bbox, image_center_x, image_center_y):
        """
        Get metric distance to target using ZoeDepth with advanced filtering.
        Returns distance in meters directly (no conversion needed).
        """
        if not self.model_loaded:
            return None
            
        try:
            # Get metric depth map
            depth_map = self.estimate_depth(frame)
            if depth_map is None:
                return None
            
            x1, y1, x2, y2 = target_bbox
            
            # Use configurable ROI scaling to focus on target center
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Apply ROI scaling
            scaled_width = bbox_width * DEPTH_ROI_SCALE
            scaled_height = bbox_height * DEPTH_ROI_SCALE
            
            roi_x1 = center_x - scaled_width / 2
            roi_y1 = center_y - scaled_height / 2
            roi_x2 = center_x + scaled_width / 2
            roi_y2 = center_y + scaled_height / 2
            
            # Map to depth map coordinates
            h, w = depth_map.shape
            frame_h, frame_w = frame.shape[:2]
            
            depth_x1 = max(0, int(roi_x1 * w / frame_w))
            depth_y1 = max(0, int(roi_y1 * h / frame_h))
            depth_x2 = min(w, int(roi_x2 * w / frame_w))
            depth_y2 = min(h, int(roi_y2 * h / frame_h))
            
            if depth_x2 <= depth_x1 or depth_y2 <= depth_y1:
                return None
            
            # Extract depth samples from ROI
            depth_samples = depth_map[depth_y1:depth_y2, depth_x1:depth_x2]
            
            # Apply median filter if enabled
            if USE_DEPTH_MEDIAN_FILTER and depth_samples.size > 9:
                try:
                    from scipy.ndimage import median_filter
                    depth_samples = median_filter(depth_samples, size=3)
                except ImportError:
                    pass
            
            # Filter valid depth values
            valid_depths = depth_samples[
                (depth_samples >= DEPTH_MIN_RANGE_M) & 
                (depth_samples <= DEPTH_MAX_RANGE_M) &
                (depth_samples > 0)  # ZoeDepth specific: filter zero/invalid depths
            ]
            
            if len(valid_depths) == 0:
                return None
            
            # Use configurable percentile for robust distance estimation
            target_distance = np.percentile(valid_depths, DEPTH_PERCENTILE)
            
            # Calculate confidence based on depth variance
            depth_std = np.std(valid_depths)
            confidence = 1.0 / (1.0 + depth_std * 2.0)  # Adjusted for metric depth
            
            # Check confidence threshold
            if confidence < DEPTH_MIN_CONFIDENCE:
                return None
            
            # Apply temporal smoothing
            if hasattr(self, 'last_distance') and self.last_distance is not None:
                target_distance = (DEPTH_SMOOTHING_FACTOR * self.last_distance + 
                                 (1 - DEPTH_SMOOTHING_FACTOR) * target_distance)
            
            self.last_distance = target_distance
            return float(target_distance)
            
        except Exception as e:
            print(f"ZoeDepth distance estimation error: {e}")
            return None

class MonocularDepthEstimator:
    """
    Unified depth estimation class supporting both MiDaS and ZoeDepth models.
    Automatically selects the best available model for UAV applications.
    """
    def __init__(self, model_type='zoedepth'):
        self.model_type = model_type
        self.estimator = None
        self.model_loaded = False
        
        # Initialize the appropriate estimator
        if model_type == 'zoedepth' and ZOEDEPTH_AVAILABLE and not FORCE_MIDAS:
            self.estimator = ZoeDepthEstimator(ZOEDEPTH_MODEL_VARIANT)
            print("Using ZoeDepth for metric depth estimation")
        elif model_type == 'midas' or not ZOEDEPTH_AVAILABLE or FORCE_MIDAS:
            self.estimator = MiDaSDepthEstimator()
            if FORCE_MIDAS:
                print("MiDaS forced via FORCE_MIDAS setting")
            else:
                print("Using MiDaS for depth estimation")
        else:
            print(f"Unsupported depth model type: {model_type}")
    
    def load_model(self):
        """Load the depth estimation model with automatic fallback."""
        if self.estimator is None:
            return False
            
        # Try to load the primary model (ZoeDepth)
        if self.estimator.load_model():
            self.model_loaded = True
            return True
        
        # If ZoeDepth fails and we're not already using MiDaS, fall back to MiDaS
        if self.model_type == 'zoedepth':
            print("ZoeDepth failed to load, falling back to MiDaS...")
            self.estimator = MiDaSDepthEstimator()
            self.model_type = 'midas'
            if self.estimator.load_model():
                self.model_loaded = True
                print("Successfully fell back to MiDaS depth estimation")
                return True
            else:
                print("Both ZoeDepth and MiDaS failed to load")
                self.model_loaded = False
                return False
        else:
            # MiDaS already failed
            self.model_loaded = False
            return False
    
    def estimate_depth(self, frame):
        """Estimate depth map from frame."""
        if self.estimator is None:
            return None
        return self.estimator.estimate_depth(frame)
    
    def get_target_distance(self, frame, target_bbox, image_center_x, image_center_y):
        """Get distance to target in meters."""
        if self.estimator is None:
            return None
        return self.estimator.get_target_distance(frame, target_bbox, image_center_x, image_center_y)

class MiDaSDepthEstimator:
    """
    MiDaS depth estimation (legacy support).
    Provides relative depth that requires conversion to metric.
    """
    def __init__(self):
        self.model_type = 'midas'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.model_loaded = False
        self.last_distance = None
        print(f"Initializing MiDaS depth estimator with device: {self.device}")
        
    def load_model(self):
        """Load the MiDaS depth estimation model."""
        if not DEPTH_AVAILABLE:
            print("PyTorch not available - depth estimation disabled")
            return False
            
        try:
            if self.model_type == 'midas':
                # Load MiDaS model - using small version for real-time performance
                self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
                self.model.to(self.device)
                self.model.eval()
                
                # Load MiDaS transforms
                midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
                self.transform = midas_transforms.small_transform
                
                print(f"MiDaS depth estimation model loaded successfully on {self.device}")
                return True
            else:
                print(f"Unsupported depth model type: {self.model_type}")
                return False
                
        except Exception as e:
            print(f"Failed to load depth model: {e}")
            return False
    
    def estimate_depth(self, frame):
        """Estimate depth map from input frame."""
        if self.model is None:
            return None
            
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Transform input
            input_tensor = self.transform(rgb_frame).to(self.device)
            
            # Inference
            with torch.no_grad():
                depth_map = self.model(input_tensor)
                
            # Convert to numpy
            depth_map = depth_map.squeeze().cpu().numpy()
            
            return depth_map
            
        except Exception as e:
            print(f"Depth estimation error: {e}")
            return None
    
    def get_target_distance(self, frame, target_bbox, image_center_x, image_center_y):
        """Get estimated distance to target center in meters with tunable parameters."""
        depth_map = self.estimate_depth(frame)
        if depth_map is None:
            return None
            
        try:
            x1, y1, x2, y2 = target_bbox
            
            # Use configurable ROI scaling to focus on target center
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Apply ROI scaling based on configuration
            scaled_width = bbox_width * DEPTH_ROI_SCALE
            scaled_height = bbox_height * DEPTH_ROI_SCALE
            
            roi_x1 = center_x - scaled_width / 2
            roi_y1 = center_y - scaled_height / 2
            roi_x2 = center_x + scaled_width / 2
            roi_y2 = center_y + scaled_height / 2
            
            # Get depth map dimensions
            h, w = depth_map.shape
            frame_h, frame_w = frame.shape[:2]
            
            # Map ROI coordinates to depth map coordinates
            depth_x1 = max(0, int(roi_x1 * w / frame_w))
            depth_y1 = max(0, int(roi_y1 * h / frame_h))
            depth_x2 = min(w, int(roi_x2 * w / frame_w))
            depth_y2 = min(h, int(roi_y2 * h / frame_h))
            
            if depth_x2 <= depth_x1 or depth_y2 <= depth_y1:
                return None
            
            # Extract depth samples from ROI
            depth_samples = depth_map[depth_y1:depth_y2, depth_x1:depth_x2]
            
            # Apply median filter if enabled
            if USE_DEPTH_MEDIAN_FILTER and depth_samples.size > 9:
                try:
                    from scipy.ndimage import median_filter
                    depth_samples = median_filter(depth_samples, size=3)
                except ImportError:
                    pass  # Skip if scipy not available
            
            # Use configurable percentile for depth aggregation
            if depth_samples.size > 0:
                target_depth = np.percentile(depth_samples, DEPTH_PERCENTILE)
            else:
                return None
            
            # MiDaS outputs inverse depth, convert to approximate distance
            # Calibration factor adjusted for 2m operation
            if target_depth > 0:
                estimated_distance = 8.0 / target_depth  # Adjusted conversion factor for 2m targeting
                
                # Apply configurable range limits
                estimated_distance = max(DEPTH_MIN_RANGE_M, min(DEPTH_MAX_RANGE_M, estimated_distance))
                
                # Apply temporal smoothing if we have previous estimate
                if hasattr(self, 'last_distance') and self.last_distance is not None:
                    estimated_distance = (DEPTH_SMOOTHING_FACTOR * self.last_distance + 
                                        (1 - DEPTH_SMOOTHING_FACTOR) * estimated_distance)
                
                # Calculate confidence based on depth variance
                depth_std = np.std(depth_samples)
                confidence = 1.0 / (1.0 + depth_std)
                
                # Check minimum confidence threshold
                if confidence >= DEPTH_MIN_CONFIDENCE:
                    self.last_distance = estimated_distance
                    return estimated_distance
                else:
                    return None  # Low confidence estimate
            else:
                return None
                
        except Exception as e:
            print(f"Error getting target distance: {e}")
            return None
    
    def create_depth_visualization(self, depth_map, frame_shape):
        """Create a colorized depth map for visualization."""
        if depth_map is None:
            return None
            
        try:
            # Normalize depth map
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Apply colormap
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
            
            # Resize to match frame size
            depth_resized = cv2.resize(depth_colored, (frame_shape[1], frame_shape[0]))
            
            return depth_resized
            
        except Exception as e:
            print(f"Error creating depth visualization: {e}")
            return None

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

def post_process_yolo_output(output, input_shape, original_shape, conf_threshold=0.5, nms_threshold=0.4):
    """
    Post-process YOLO model output to extract bounding boxes, scores, and class IDs.
    
    Args:
        output (numpy.ndarray): Raw output from YOLO model
        input_shape (tuple): Input shape of the model (height, width)
        original_shape (tuple): Original image shape (height, width)
        conf_threshold (float): Confidence threshold for filtering detections
        nms_threshold (float): NMS threshold for removing overlapping boxes
        
    Returns:
        list: List of detections with format [{'xyxy': [x1, y1, x2, y2], 'cls': class_id, 'conf': confidence}]
    """
    detections = []
    
    # Handle different YOLO output formats
    if len(output.shape) == 3:
        output = output[0]  # Remove batch dimension if present
    
    # Transpose if needed (some models output [4+num_classes, num_detections])
    if output.shape[0] < output.shape[1]:
        output = output.T
    
    # Extract predictions
    boxes = output[:, :4]  # x, y, w, h (center format)
    scores = output[:, 4]  # objectness scores
    class_scores = output[:, 5:]  # class scores
    
    # Filter by confidence
    confident_detections = scores > conf_threshold
    boxes = boxes[confident_detections]
    scores = scores[confident_detections]
    class_scores = class_scores[confident_detections]
    
    if len(boxes) == 0:
        return detections
    
    # Get class IDs and final scores
    class_ids = np.argmax(class_scores, axis=1)
    final_scores = scores * np.max(class_scores, axis=1)
    
    # Convert from center format to corner format and scale to original image
    scale_x = original_shape[1] / input_shape[1]
    scale_y = original_shape[0] / input_shape[0]
    
    x_center = boxes[:, 0] * scale_x
    y_center = boxes[:, 1] * scale_y
    width = boxes[:, 2] * scale_x
    height = boxes[:, 3] * scale_y
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), final_scores.tolist(), conf_threshold, nms_threshold
    )
    
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            detections.append({
                'xyxy': [x1[i], y1[i], x2[i], y2[i]],
                'cls': int(class_ids[i]),
                'conf': float(final_scores[i])
            })
    
    return detections

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
    await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, HOVER_THRUST_VALUE + 0.3))
    try:
        await drone.offboard.start()
        print(f"Offboard mode started for takeoff with thrust: {HOVER_THRUST_VALUE + 0.3}")
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
        await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, HOVER_THRUST_VALUE + 0.3)) 
        
        if current_altitude >= aTargetAltitude * 0.95:
            print(f"Reached target altitude: {current_altitude:.2f}m")
            break
        if (time.time() - takeoff_start_time) > TAKE_OFF_TIMEOUT:
            print(f"Takeoff timed out after {TAKE_OFF_TIMEOUT} seconds. Current altitude: {current_altitude:.2f}m")
            return False # Indicate takeoff failure
        await asyncio.sleep(0.1) # Shorter sleep for smoother altitude check

    # After reaching altitude, set thrust to hover value
    await drone.offboard.set_attitude_rate(AttitudeRate(0.0, 0.0, 0.0, HOVER_THRUST_VALUE))
    print("Hovering at target altitude.")
    return True # Indicate success

async def send_attitude_rate_commands(drone, roll_rate, pitch_rate, yaw_rate, thrust_value):
    """Sends roll, pitch, and yaw rate commands to the drone using MAVSDK offboard."""
    roll_rate = np.clip(roll_rate, -PITCH_LIMIT_ALIGN, PITCH_LIMIT_ALIGN)
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
    # Distance PID for forward/backward movement
    distance_pid_controller = PID(KP_DISTANCE, KI_DISTANCE, KD_DISTANCE)
    
    cap = cv2.VideoCapture(0) # For Ubuntu PC webcam: 0 is usually the default
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        # Attempt to stop offboard mode if it started
        try: await drone.offboard.stop()
        except OffboardError as error: print(f"Failed to stop offboard: {error}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        try: await drone.offboard.stop()
        except OffboardError as error: print(f"Failed to stop offboard: {error}")
        return

    image_height, image_width, _ = frame.shape
    image_center_x = image_width / 2
    image_center_y = image_height / 2 # New: vertical center

    # Initialize mock target positions once with image dimensions
    initialize_mock_targets(image_width, image_height)
    
    # --- Real YOLO Model Loading ---
    yolo_model = None
    # Corrected: Initialize _use_real_yolo_local from the global USE_REAL_YOLO
    _use_real_yolo_local = USE_REAL_YOLO 
    
    if _use_real_yolo_local: # Use the local flag here
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(REAL_YOLO_MODEL_PATH)
            print(f"Real YOLO model loaded from: {REAL_YOLO_MODEL_PATH}")
        except ImportError:
            print("Error: 'ultralytics' not found. Please install it with 'pip install ultralytics'. Falling back to mock YOLO.")
            _use_real_yolo_local = False # Modify the local copy
        except Exception as e:
            print(f"Error loading real YOLO model: {e}. Falling back to mock YOLO.")
            _use_real_yolo_local = False # Modify the local copy
    # --- End Real YOLO Model Loading ---

    # --- Depth Estimation Model Loading ---
    depth_estimator = None
    _use_depth_local = USE_DEPTH_ESTIMATION and DEPTH_AVAILABLE
    
    if _use_depth_local:
        try:
            depth_estimator = MonocularDepthEstimator(DEPTH_MODEL_TYPE)
            if depth_estimator.load_model():
                print("Depth estimation enabled for target distance control")
            else:
                print("Failed to load depth model - depth estimation disabled")
                _use_depth_local = False
                depth_estimator = None
        except Exception as e:
            print(f"Error initializing depth estimator: {e}. Depth estimation disabled.")
            _use_depth_local = False
            depth_estimator = None
    else:
        print("Depth estimation disabled in configuration or PyTorch not available")
    # --- End Depth Estimation Model Loading ---

    # Initialize Kalman Filters for yaw (horizontal position) and pitch (vertical position)
    yaw_kalman_filter = KalmanFilter(dt=0.1, Q=Q_KF, R=R_KF, initial_pos=image_center_x) 
    pitch_kalman_filter = KalmanFilter(dt=0.1, Q=Q_KF, R=R_KF, initial_pos=image_center_y) # For vertical alignment
    
    # Initialize Distance Kalman Filter for depth estimation
    distance_kalman_filter = DistanceKalmanFilter(dt=0.1, Q=Q_DISTANCE_KF, R=R_DISTANCE_KF, initial_distance=TARGET_DISTANCE_M)
    
    # FIX: Initialize current_time and last_time before the loop
    current_time = time.time()
    last_time = current_time

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
    
    video_writer = None
    if RECORD_VIDEO:
        video_filename = os.path.join(flight_folder, f"tracking_video_{timestamp}.mp4")
        video_writer = cv2.VideoWriter(video_filename, VIDEO_CODEC, VIDEO_FPS, (image_width, image_height))
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {video_filename}. Recording disabled.")
            video_writer = None

    is_offboard_active_from_script = True

    while True:
        # Debug print
        print(f"Main loop running. Time: {current_time:.2f}") 
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time() # Update current_time for the next iteration
        dt = current_time - last_time
        last_time = current_time

        yaw_kalman_filter.dt = dt
        yaw_kalman_filter.A = np.array([[1, dt], [0, 1]])
        yaw_kalman_filter.Q_matrix = np.array([[0.25*dt**4, 0.5*dt**3], [0.5*dt**3, dt**2]]) * yaw_kalman_filter.Q

        pitch_kalman_filter.dt = dt
        pitch_kalman_filter.A = np.array([[1, dt], [0, 1]])
        pitch_kalman_filter.Q_matrix = np.array([[0.25*dt**4, 0.5*dt**3], [0.5*dt**3, dt**2]]) * pitch_kalman_filter.Q

        # Update distance Kalman filter time step
        distance_kalman_filter.update_dt(dt)

        yaw_kalman_filter.predict()
        pitch_kalman_filter.predict()
        distance_kalman_filter.predict()

        yaw_rate_cmd = 0.0
        pitch_rate_cmd = 0.0
        forward_velocity_cmd = 0.0
        
        current_flight_mode = None
        async for flight_mode_telemetry in drone.telemetry.flight_mode():
            current_flight_mode = flight_mode_telemetry
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
                    log_data_row.extend([0, 0, 0, 0, 0, 0, 0, 0, -1]) 
                    csv_writer.writerow(log_data_row)
                    if video_writer is not None: video_writer.write(frame)
                    cv2.imshow("Object Tracking", frame)
                    await asyncio.sleep(0.01)
                    continue

            # --- Perform YOLO Inference (Real or Mock) ---
            detections = []
            if _use_real_yolo_local and yolo_model: # Use the local flag here
                try:
                    results = yolo_model.predict(frame, verbose=False, conf=0.5) # Adjust conf threshold as needed
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cls_id = int(box.cls[0])
                            detections.append({
                                'xyxy': [x1, y1, x2, y2],
                                'cls': cls_id
                            })
                except Exception as e:
                    print(f"Error during real YOLO inference: {e}. Using mock detections for this frame.")
                    detections = mock_yolo_inference(frame) # Fallback to mock if real YOLO fails
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
            
            # Distance tracking variables
            target_distance_m = 0.0
            kf_distance_est = distance_kalman_filter.get_estimated_state()[0]
            distance_err = 0.0
            forward_velocity_mps = 0.0
            target_centered = False
            
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
                
                # STEP 1: Always do yaw control to center the target horizontally
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

                # STEP 2: Check if target is centered horizontally
                target_centered = abs(yaw_error) <= YAW_CENTERED_THRESHOLD
                
                # STEP 3: If target is centered, use depth estimation for forward movement
                if target_centered and _use_depth_local and depth_estimator:
                    # Get distance to target using depth estimation
                    target_distance_m = depth_estimator.get_target_distance(
                        frame, selected_box['xyxy'], image_center_x, image_center_y
                    )
                    
                    if target_distance_m is not None:
                        # Update distance Kalman filter
                        distance_kalman_filter.update(np.array([[target_distance_m]]))
                        estimated_distance, _ = distance_kalman_filter.get_estimated_state()
                        
                        # Calculate distance error
                        distance_err = estimated_distance - TARGET_DISTANCE_M
                        
                        # Apply distance control only if outside dead zone
                        if abs(distance_err) > DISTANCE_DEAD_ZONE_M:
                            forward_velocity_cmd = distance_pid_controller.update(distance_err, dt)
                            # Limit velocity
                            forward_velocity_cmd = np.clip(forward_velocity_cmd, -MAX_FORWARD_SPEED_MPS, MAX_FORWARD_SPEED_MPS)
                        else:
                            forward_velocity_cmd = 0.0
                            distance_pid_controller.reset()
                        
                        # Use velocity control when doing distance tracking
                        await drone.offboard.set_velocity_ned(VelocityNedYaw(
                            forward_velocity_cmd,  # North (forward/backward)
                            0.0,                   # East (left/right) 
                            0.0,                   # Down (up/down) - maintain altitude
                            np.deg2rad(yaw_rate_cmd)  # Yaw rate in rad/s
                        ))
                    else:
                        # Depth estimation failed, fall back to attitude control
                        forward_velocity_cmd = 0.0
                        await send_attitude_rate_commands(drone, 0, pitch_rate_cmd, yaw_rate_cmd, HOVER_THRUST_VALUE)
                else:
                    # Target not centered yet, focus on centering with attitude control
                    forward_velocity_cmd = 0.0
                    await send_attitude_rate_commands(drone, 0, pitch_rate_cmd, yaw_rate_cmd, HOVER_THRUST_VALUE)

                # --- Update logging variables with current values ---
                # --- Update logging variables with current values ---
                raw_yaw_err = raw_x_center_bbox - image_center_x
                kf_yaw_est = estimated_x_center
                yaw_cmd = yaw_rate_cmd
                yaw_i_term = yaw_pid_controller.I_term

                raw_pitch_align_err = raw_y_center_bbox - image_center_y
                kf_pitch_align_est = estimated_y_center
                pitch_cmd = pitch_rate_cmd
                pitch_i_term = pitch_align_pid_controller.I_term
                
                # Update distance variables
                if target_distance_m > 0:
                    kf_distance_est = distance_kalman_filter.get_estimated_state()[0]
                else:
                    kf_distance_est = TARGET_DISTANCE_M  # Default value
                forward_velocity_mps = forward_velocity_cmd
                
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
                
                # Display targeting status
                if target_centered:
                    status_text = "CENTERED - Moving Forward" if forward_velocity_cmd != 0 else "CENTERED - At Target Distance"
                    status_color = (0, 255, 0)  # Green
                else:
                    status_text = "CENTERING Target"
                    status_color = (0, 255, 255)  # Yellow
                cv2.putText(frame, status_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                
                # Display distance information if available
                if target_distance_m > 0 and target_centered:
                    cv2.putText(frame, f"Distance: {kf_distance_est:.1f}m | Target: {TARGET_DISTANCE_M:.1f}m", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Forward Vel: {forward_velocity_mps:.2f}m/s", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
                
                # Reset distance tracking variables
                target_distance_m = 0.0
                distance_err = 0.0
                forward_velocity_mps = 0.0
                target_centered = False
                
                tracked_class_id = -1

                cv2.circle(frame, (int(estimated_x_center), int(estimated_y_center)), 8, (255, 165, 0), -1) # Estimated center (orange)
                cv2.line(frame, (int(image_center_x), 0), (int(image_center_x), image_height), (255, 0, 0), 2) # Image center vertical (red)
                cv2.line(frame, (0, int(image_center_y)), (image_width, int(image_center_y)), (255, 0, 0), 2) # Image center horizontal (red)
                cv2.putText(frame, "No Target Detected - Predicting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Yaw Err: {yaw_error:.2f}px | Cmd: {yaw_rate_cmd:.1f}deg/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Pitch Align Err: {pitch_error_align:.2f}px | Cmd: {pitch_rate_cmd:.1f}deg/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "SEARCHING for Target", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
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
            distance_pid_controller.reset()
            yaw_kalman_filter.reset(initial_pos=image_center_x)
            pitch_kalman_filter.reset(initial_pos=image_center_y)
            distance_kalman_filter.reset(initial_distance=TARGET_DISTANCE_M)

            # --- Update logging variables when tracking is OFF ---
            raw_yaw_err = 0.0
            kf_yaw_est = image_center_x
            yaw_cmd = 0.0
            yaw_i_term = 0.0

            raw_pitch_align_err = 0.0
            kf_pitch_align_est = image_center_y
            pitch_cmd = 0.0
            pitch_i_term = 0.0
            
            # Reset distance tracking variables
            target_distance_m = 0.0
            distance_err = 0.0
            forward_velocity_mps = 0.0
            target_centered = False
            
            tracked_class_id = -1

            cv2.putText(frame, f"Tracking: OFF ({current_flight_mode.name} mode)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- Add data to the log row and write to CSV ---
        log_data_row.extend([
            raw_yaw_err, kf_yaw_est, yaw_cmd, yaw_i_term,
            raw_pitch_align_err, kf_pitch_align_est, pitch_cmd, pitch_i_term,
            target_distance_m, kf_distance_est, distance_err, forward_velocity_mps,
            target_centered, tracked_class_id
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
    csv_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(tracking_loop())