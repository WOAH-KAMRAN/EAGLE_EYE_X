# UAV Target Tracking System - FLIGHT READY VERSION
# Safety features: RC override, altitude hold, emergency stop, timeouts, pre-flight checks
# For Jetson Orin Nano with real drone hardware

import time
import cv2
import numpy as np
import datetime
import os
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, AttitudeRate, VelocityNedYaw
from mavsdk.telemetry import FlightMode, RcStatus
import csv
import signal
import sys

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

# --- Configuration ---
# IMPORTANT: Update these for your hardware!
CONNECTION_STRING = 'serial:///dev/ttyACM0:921600'  # Or ttyTHS1 for Jetson UART
YOLO_MODEL_PATH = 'yolov8n.pt'  # <--- UPDATE THIS!

# Detection settings
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
CLASS_PRIORITIES = {
    0: 1,  # person - adjust as needed
    # Add your classes here
}

# Control limits
YAW_LIMIT = 30  # Reduced for safety
PITCH_LIMIT = 15  # Reduced for safety
ROLL_LIMIT = 15
HOVER_THRUST = 0.5  # Tune for your drone weight
TAKEOFF_THRUST = 0.7  # Tune for smooth takeoff

# Dead zones (pixels)
DEAD_ZONE_YAW = 20
DEAD_ZONE_PITCH = 20

# Altitude control
TARGET_ALTITUDE = 2.0  # meters
ALTITUDE_TOLERANCE = 0.3  # meters
MAX_ALTITUDE = 10.0  # Safety limit
MIN_ALTITUDE = 0.5  # Don't go below this

# PID Gains - TUNE THESE IN SIMULATOR FIRST!
KP_YAW = 0.03
KI_YAW = 0.001
KD_YAW = 0.01

KP_PITCH = 0.02
KI_PITCH = 0.001
KD_PITCH = 0.01

KP_ALT = 0.5
KI_ALT = 0.05
KD_ALT = 0.1

# Kalman Filter
Q_KF = 0.1
R_KF = 10

# Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_INDEX = 0  # USB camera, or use GStreamer string

# Video recording
RECORD_VIDEO = True
VIDEO_FOLDER = "flights"

# Safety timeouts (seconds)
TELEMETRY_TIMEOUT = 2.0
DETECTION_TIMEOUT = 5.0  # Max time without detection before hover
TAKEOFF_TIMEOUT = 20.0

# Global emergency stop flag
EMERGENCY_STOP = False

def signal_handler(sig, frame):
    """Handle Ctrl+C for emergency stop"""
    global EMERGENCY_STOP
    print("\n⚠️  EMERGENCY STOP TRIGGERED!")
    EMERGENCY_STOP = True

signal.signal(signal.SIGINT, signal_handler)

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

# --- Kalman Filter ---
class KalmanFilter:
    def __init__(self, Q=0.1, R=10, initial_pos=0.0):
        self.Q = Q
        self.R = R
        self.x = np.array([[initial_pos], [0.0]])  # [position, velocity]
        self.P = np.array([[1000.0, 0.0], [0.0, 1000.0]])
        self.F = np.array([[1, 0.033], [0, 1]])  # State transition (30 FPS)
        self.H = np.array([[1, 0]])
        self.Q_matrix = np.array([[0.25, 0.5], [0.5, 1.0]]) * Q
        self.R_matrix = np.array([[R]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q_matrix

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

# --- YOLO Detector ---
class YOLODetector:
    def __init__(self, model_path):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        if torch.cuda.is_available():
            self.device = 'cuda'
            print("✓ Using CUDA")
        else:
            self.device = 'cpu'
            print("⚠ Using CPU (slower)")
        
        self.model.to(self.device)
        
        # Warm up
        dummy = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        _ = self.model.predict(dummy, verbose=False, conf=0.5)
        print("✓ YOLO ready")
    
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

# --- Telemetry Helper with Timeout ---
async def get_telemetry(telemetry_stream, timeout=TELEMETRY_TIMEOUT):
    """Get telemetry with timeout protection"""
    try:
        return await asyncio.wait_for(
            telemetry_stream.__anext__(),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"Telemetry timeout after {timeout}s")

# --- Pre-flight Checks ---
async def preflight_checks(drone):
    """Basic companion computer checks - FC handles sensor checks"""
    print("\n🔍 Running companion computer checks...")
    
    # 1. Check connection
    print("  Checking drone connection...")
    try:
        health = await asyncio.wait_for(
            drone.telemetry.health().__anext__(),
            timeout=5.0
        )
        print("  ✓ Telemetry connection OK")
    except asyncio.TimeoutError:
        print("  ✗ Cannot get telemetry")
        return False
    
    # 2. Check RC (important for safety override)
    print("  Checking RC connection...")
    try:
        rc_status = await asyncio.wait_for(
            drone.telemetry.rc_status().__anext__(),
            timeout=5.0
        )
        if not rc_status.is_available:
            print("  ✗ RC not connected - needed for safety override")
            return False
        else:
            print(f"  ✓ RC OK (signal: {rc_status.signal_strength_percent}%)")
    except asyncio.TimeoutError:
        print("  ⚠ RC check timeout")
        return False
    
    print("✓ Companion computer checks passed")
    print("⚠️  Ensure FC pre-arm checks pass before arming\n")
    return True

# --- Arm and Takeoff ---
async def arm_and_takeoff(drone, target_alt):
    """Safe arming and takeoff procedure"""
    print(f"\n🚁 Starting takeoff to {target_alt}m...")
    
    # Set to HOLD mode
    print("  Setting HOLD mode...")
    await asyncio.sleep(1)
    try:
        await drone.action.hold()
        await asyncio.sleep(1)
    except Exception as e:
        print(f"  ⚠ HOLD mode failed: {e}")
    
    # Arm
    print("  Arming motors...")
    try:
        await drone.action.arm()
        await asyncio.sleep(2)
        print("  ✓ Armed")
    except Exception as e:
        print(f"  ✗ Arming failed: {e}")
        return False
    
    # Start offboard
    await drone.offboard.set_attitude_rate(AttitudeRate(0, 0, 0, TAKEOFF_THRUST))
    try:
        await drone.offboard.start()
        print("  ✓ Offboard mode started")
    except OffboardError as e:
        print(f"  ✗ Offboard failed: {e}")
        return False
    
    # Takeoff
    print("  Taking off...")
    start_time = time.time()
    
    while time.time() - start_time < TAKEOFF_TIMEOUT:
        if EMERGENCY_STOP:
            return False
        
        try:
            pos = await asyncio.wait_for(
                drone.telemetry.position().__anext__(),
                timeout=2.0
            )
            alt = pos.relative_altitude_m
        except:
            print("  ⚠ Telemetry lost during takeoff")
            return False
        
        print(f"    Altitude: {alt:.2f}m / {target_alt:.2f}m")
        
        if alt >= target_alt * 0.95:
            print(f"  ✓ Reached {alt:.2f}m")
            break
        
        if alt >= MAX_ALTITUDE:
            print(f"  ⚠ Max altitude reached")
            break
        
        await drone.offboard.set_attitude_rate(AttitudeRate(0, 0, 0, TAKEOFF_THRUST))
        await asyncio.sleep(0.2)
    else:
        print("  ✗ Takeoff timeout")
        return False
    
    # Switch to hover
    await drone.offboard.set_attitude_rate(AttitudeRate(0, 0, 0, HOVER_THRUST))
    print("  ✓ Hovering\n")
    return True

# --- Main Tracking Loop ---
async def tracking_loop():
    """Main flight and tracking loop with safety features"""
    global EMERGENCY_STOP
    
    print("=" * 60)
    print("UAV TARGET TRACKING - FLIGHT READY")
    print("=" * 60)
    
    # Connect to drone
    drone = System()
    print(f"\nConnecting to drone: {CONNECTION_STRING}")
    await drone.connect(system_address=CONNECTION_STRING)
    
    print("Waiting for drone...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✓ Drone connected\n")
            break
        await asyncio.sleep(1)
    
    # Pre-flight checks
    if not await preflight_checks(drone):
        print("❌ Pre-flight checks failed. Aborting.")
        return
    
    # Wait for GUIDED/OFFBOARD mode
    print("📡 Waiting for GUIDED/OFFBOARD mode...")
    print("   Switch to GUIDED mode in your ground station\n")
    while True:
        try:
            fm = await asyncio.wait_for(
                drone.telemetry.flight_mode().__anext__(),
                timeout=5.0
            )
            if fm == FlightMode.OFFBOARD:
                print("✓ Ready for offboard control\n")
                break
            print(f"   Current mode: {fm.name}")
            await asyncio.sleep(2)
        except asyncio.TimeoutError:
            print("⚠ Mode check timeout")
            return
    
    # Initialize camera
    print("📷 Initializing camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    if not cap.isOpened():
        print("✗ Camera failed to open")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("✗ Cannot read from camera")
        cap.release()
        return
    
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    print(f"✓ Camera ready ({w}x{h})\n")
    
    # Initialize YOLO
    try:
        detector = YOLODetector(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"✗ YOLO failed: {e}")
        cap.release()
        return
    
    # Takeoff
    if not await arm_and_takeoff(drone, TARGET_ALTITUDE):
        print("✗ Takeoff failed")
        cap.release()
        return
    
    # Initialize controllers
    yaw_pid = PID(KP_YAW, KI_YAW, KD_YAW, output_limit=YAW_LIMIT)
    pitch_pid = PID(KP_PITCH, KI_PITCH, KD_PITCH, output_limit=PITCH_LIMIT)
    alt_pid = PID(KP_ALT, KI_ALT, KD_ALT, output_limit=0.3)
    
    yaw_kf = KalmanFilter(Q=Q_KF, R=R_KF, initial_pos=cx)
    pitch_kf = KalmanFilter(Q=Q_KF, R=R_KF, initial_pos=cy)
    
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    flight_dir = os.path.join(VIDEO_FOLDER, f"flight_{timestamp}")
    os.makedirs(flight_dir, exist_ok=True)
    
    log_file = open(os.path.join(flight_dir, f"log_{timestamp}.csv"), 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Time', 'Mode', 'Alt', 'Alt_Err', 'Yaw_Cmd', 'Pitch_Cmd', 
                        'Class', 'Conf', 'RC_OK'])
    
    video_writer = None
    if RECORD_VIDEO:
        vid_path = os.path.join(flight_dir, f"video_{timestamp}.mp4")
        video_writer = cv2.VideoWriter(
            vid_path, 
            cv2.VideoWriter_fourcc(*'mp4v'),
            CAMERA_FPS, 
            (w, h)
        )
    
    print("🎯 Starting tracking...\n")
    print("Controls:")
    print("  Q or Ctrl+C: Emergency stop and land")
    print("  Switch to LOITER/HOLD in GCS to pause tracking\n")
    
    is_offboard = True
    last_detection_time = time.time()
    frame_count = 0
    
    try:
        while not EMERGENCY_STOP:
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                print("⚠ Camera read failed")
                break
            
            # Get telemetry with timeout protection
            try:
                mode = await asyncio.wait_for(
                    drone.telemetry.flight_mode().__anext__(),
                    timeout=TELEMETRY_TIMEOUT
                )
                
                pos = await asyncio.wait_for(
                    drone.telemetry.position().__anext__(),
                    timeout=TELEMETRY_TIMEOUT
                )
                altitude = pos.relative_altitude_m
                
                rc = await asyncio.wait_for(
                    drone.telemetry.rc_status().__anext__(),
                    timeout=TELEMETRY_TIMEOUT
                )
                rc_ok = rc.is_available
                
            except asyncio.TimeoutError:
                print("⚠ Telemetry timeout - landing for safety")
                break
            except Exception as e:
                print(f"⚠ Telemetry error: {e}")
                break
            
            # Check RC override
            if not rc_ok:
                print("⚠ RC connection lost - landing")
                break
            
            # Altitude limits
            if altitude > MAX_ALTITUDE:
                print(f"⚠ Max altitude {MAX_ALTITUDE}m exceeded")
                break
            if altitude < MIN_ALTITUDE:
                print(f"⚠ Below minimum altitude {MIN_ALTITUDE}m")
                break
            
            yaw_cmd = pitch_cmd = 0.0
            tracked_class = -1
            confidence = 0.0
            
            if mode == FlightMode.OFFBOARD:
                if not is_offboard:
                    try:
                        await drone.offboard.start()
                        is_offboard = True
                        yaw_pid.reset()
                        pitch_pid.reset()
                        alt_pid.reset()
                    except OffboardError:
                        pass
                
                # Run detection every frame
                detections = detector.detect(frame, CONF_THRESHOLD, IOU_THRESHOLD)
                
                # Predict Kalman
                yaw_kf.predict()
                pitch_kf.predict()
                
                # Select best target
                best_det = None
                best_priority = -1
                for det in detections:
                    priority = CLASS_PRIORITIES.get(det['cls'], 0)
                    if priority > best_priority:
                        best_priority = priority
                        best_det = det
                
                if best_det:
                    last_detection_time = time.time()
                    
                    x1, y1, x2, y2 = best_det['xyxy']
                    raw_x = (x1 + x2) / 2
                    raw_y = (y1 + y2) / 2
                    
                    # Update Kalman
                    yaw_kf.update(np.array([[raw_x]]))
                    pitch_kf.update(np.array([[raw_y]]))
                    
                    est_x, _ = yaw_kf.get_state()
                    est_y, _ = pitch_kf.get_state()
                    
                    # Compute errors
                    yaw_err = est_x - cx
                    pitch_err = est_y - cy
                    
                    # PID control
                    if abs(yaw_err) > DEAD_ZONE_YAW:
                        yaw_cmd = yaw_pid.update(yaw_err)
                    else:
                        yaw_pid.reset()
                    
                    if abs(pitch_err) > DEAD_ZONE_PITCH:
                        pitch_cmd = pitch_pid.update(pitch_err)
                    else:
                        pitch_pid.reset()
                    
                    tracked_class = best_det['cls']
                    confidence = best_det['conf']
                    
                    # Draw detections
                    for det in detections:
                        dx1, dy1, dx2, dy2 = [int(x) for x in det['xyxy']]
                        color = (0, 255, 0) if det == best_det else (100, 100, 255)
                        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
                        label = f"{det['cls']}:{det['conf']:.2f}"
                        cv2.putText(frame, label, (dx1, dy1-5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    cv2.circle(frame, (int(raw_x), int(raw_y)), 5, (0, 0, 255), -1)
                    cv2.circle(frame, (int(est_x), int(est_y)), 8, (255, 0, 255), 2)
                    
                else:
                    # No detection - check timeout
                    if time.time() - last_detection_time > DETECTION_TIMEOUT:
                        # Lost target - hover in place
                        yaw_cmd = pitch_cmd = 0.0
                        yaw_pid.reset()
                        pitch_pid.reset()
                    else:
                        # Use last Kalman estimate
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
                
                # Send commands
                await drone.offboard.set_attitude_rate(
                    AttitudeRate(
                        0.0,  # roll
                        np.deg2rad(-pitch_cmd),
                        np.deg2rad(yaw_cmd),
                        thrust
                    )
                )
                
                # Draw UI
                cv2.line(frame, (int(cx), 0), (int(cx), h), (255, 0, 0), 1)
                cv2.line(frame, (0, int(cy)), (w, int(cy)), (255, 0, 0), 1)
                
                status_color = (0, 255, 0) if best_det else (0, 165, 255)
                cv2.putText(frame, f"TRACKING: {tracked_class}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(frame, f"Alt: {altitude:.1f}m (T:{TARGET_ALTITUDE:.1f}m)", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Yaw: {yaw_cmd:.1f} Pitch: {pitch_cmd:.1f}", 
                          (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # RC status
                rc_color = (0, 255, 0) if rc_ok else (0, 0, 255)
                cv2.putText(frame, f"RC: {'OK' if rc_ok else 'LOST'}", 
                          (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rc_color, 2)
            
            else:
                # Not in offboard mode
                if is_offboard:
                    try:
                        await drone.offboard.stop()
                        is_offboard = False
                    except:
                        pass
                
                yaw_pid.reset()
                pitch_pid.reset()
                alt_pid.reset()
                yaw_kf.reset(cx)
                pitch_kf.reset(cy)
                
                cv2.putText(frame, f"STANDBY ({mode.name})", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                cv2.putText(frame, "Switch to GUIDED to resume", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Log data
            log_writer.writerow([
                time.time(), mode.name, altitude, 
                TARGET_ALTITUDE - altitude,
                yaw_cmd, pitch_cmd, tracked_class, confidence, rc_ok
            ])
            
            if video_writer:
                video_writer.write(frame)
            
            cv2.imshow("UAV Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⚠️  Stop requested via 'Q' key")
                break
    
    except Exception as e:
        print(f"\n❌ Error in tracking loop: {e}")
    
    finally:
        # Safe landing procedure
        print("\n🛬 Landing sequence initiated...")
        
        if is_offboard:
            try:
                await drone.offboard.stop()
                print("  ✓ Offboard stopped")
            except:
                pass
        
        try:
            print("  Landing...")
            await drone.action.land()
            await asyncio.sleep(8)
            
            print("  Disarming...")
            await drone.action.disarm()
            print("  ✓ Disarmed")
        except Exception as e:
            print(f"  ⚠ Landing error: {e}")
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        log_file.close()
        cv2.destroyAllWindows()
        
        print("\n✓ Flight complete")
        print(f"📁 Data saved to: {flight_dir}\n")

if __name__ == "__main__":
    try:
        asyncio.run(tracking_loop())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")