#!/usr/bin/env python3
"""
UAV Target Tracking System with TensorRT and Following Capability
Supports: Jetson Orin Nano, RPi Cam v2, PC Webcam, MAVSDK control
Features: Target prioritization, Kalman filtering, PID control, forward/backward following
"""

import time, cv2, numpy as np, datetime, os, asyncio, csv, random
from mavsdk import System
from mavsdk.offboard import OffboardError, AttitudeRate, VelocityNedYaw
from mavsdk.telemetry import FlightMode

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

# ===== CONFIGURATION =====
CONNECTION_STRING = 'udpin://127.0.0.1:14550'
USE_REAL_YOLO = False
REAL_YOLO_MODEL_PATH = '/path/to/your/yolov8n.engine'
CLASS_PRIORITIES = {2: 4, 1: 3, 3: 2, 0: 1}

# Control limits
YAW_LIMIT = 50
PITCH_LIMIT_ALIGN = 20
TAKEOFF_THRUST_VALUE = 0.95
HOVER_THRUST_VALUE = 0.5
ALTITUDE_CAP_M = 10.0
DEAD_ZONE_YAW = 15
DEAD_ZONE_PITCH_ALIGN = 15

# Following configuration
ENABLE_FOLLOWING = True
TARGET_BOX_SIZE_RATIO = 0.20
DEAD_ZONE_FOLLOW = 0.05
MAX_FORWARD_VELOCITY = 2.0
MAX_BACKWARD_VELOCITY = -1.5

# PID gains
KP_YAW, KI_YAW, KD_YAW = 0.5, 0.01, 0.2
KP_PITCH_ALIGN, KI_PITCH_ALIGN, KD_PITCH_ALIGN = 0.05, 0.001, 0.01
KP_FOLLOW, KI_FOLLOW, KD_FOLLOW = 1.5, 0.05, 0.3

# Kalman filter gains
Q_KF, R_KF = 0.1, 10

# Camera configuration
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 640, 480, 30
USE_GSTREAMER_CAMERA = False
GSTREAMER_PIPELINE = f"nvarguscamsrc sensor_id=0 ! video/x-raw(memory:NVMM), width=(int){CAMERA_WIDTH}, height=(int){CAMERA_HEIGHT}, format=(string)NV12, framerate=(fraction){CAMERA_FPS}/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int){CAMERA_WIDTH}, height=(int){CAMERA_HEIGHT}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

# Video recording
RECORD_VIDEO = True
VIDEO_FOLDER = "flights"
VIDEO_FPS = CAMERA_FPS
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')

# ===== CLASSES =====
class TensorRTInference:
    def __init__(self, engine_file_path):
        self.engine_file_path = engine_file_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        self.input_shape, self.output_shapes = None, []
        
    def load_engine(self):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        try:
            with open(self.engine_file_path, 'rb') as f:
                self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
            if not self.engine:
                raise RuntimeError(f"Failed to load {self.engine_file_path}")
            self.context = self.engine.create_execution_context()
            self._setup_buffers()
            print(f"TensorRT loaded: {self.input_shape}")
            return True
        except Exception as e:
            print(f"TensorRT error: {e}")
            return False
    
    def _setup_buffers(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
                self.input_shape = self.engine.get_binding_shape(binding)
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                self.output_shapes.append(self.engine.get_binding_shape(binding))
    
    def infer(self, input_data):
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
            outputs.append(output['host'].copy())
        self.stream.synchronize()
        return [output.reshape(shape) for output, shape in zip(outputs, self.output_shapes)]
    
    def get_input_shape(self):
        return self.input_shape

class PID:
    def __init__(self, Kp, Ki, Kd, integrator_max=500):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.P_term = self.I_term = self.D_term = self.last_error = 0
        self.integrator_max = integrator_max

    def update(self, error, dt):
        self.P_term = self.Kp * error
        self.I_term += self.Ki * error * dt
        self.I_term = np.clip(self.I_term, -self.integrator_max, self.integrator_max)
        self.D_term = self.Kd * ((error - self.last_error) / dt) if dt > 0 else 0
        self.last_error = error
        return self.P_term + self.I_term + self.D_term
    
    def reset(self):
        self.P_term = self.I_term = self.D_term = self.last_error = 0

class KalmanFilter:
    def __init__(self, dt, Q, R, initial_pos=0.0):
        self.dt, self.Q, self.R = dt, Q, R
        self.x = np.array([[initial_pos], [0.0]])
        self.A = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q_matrix = np.array([[0.25*dt**4, 0.5*dt**3], [0.5*dt**3, dt**2]]) * Q
        self.R_matrix = np.array([[R]])
        self.P = np.array([[1000.0, 0.0], [0.0, 1000.0]])

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q_matrix

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R_matrix
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P -= K @ self.H @ self.P

    def get_estimated_state(self):
        return self.x[0, 0], self.x[1, 0]
    
    def reset(self, initial_pos=0.0):
        self.x = np.array([[initial_pos], [0.0]])
        self.P = np.array([[1000.0, 0.0], [0.0, 1000.0]])

# ===== MOCK DETECTION =====
mock_target_pos = {i: {'x': 0, 'y': 0} for i in range(4)}

def initialize_mock_targets(w, h):
    mock_target_pos[0] = {'x': w*0.2, 'y': h*0.3}
    mock_target_pos[1] = {'x': w*0.5, 'y': h*0.5}
    mock_target_pos[2] = {'x': w*0.8, 'y': h*0.7}
    mock_target_pos[3] = {'x': w*0.3, 'y': h*0.6}

def mock_yolo_inference(frame):
    h, w, _ = frame.shape
    results = []
    for cid in mock_target_pos:
        mock_target_pos[cid]['x'] = np.clip(mock_target_pos[cid]['x'] + random.uniform(-5, 5), w*0.1, w*0.9)
        mock_target_pos[cid]['y'] = np.clip(mock_target_pos[cid]['y'] + random.uniform(-3, 3), h*0.1, h*0.9)
    
    def create_det(x, y, cid):
        x += random.uniform(-10, 10)
        y += random.uniform(-5, 5)
        bw, bh = 80 + random.uniform(-10, 10), 80 + random.uniform(-10, 10)
        return {'xyxy': [x-bw/2, y-bh/2, x+bw/2, y+bh/2], 'cls': cid}
    
    if random.random() < 0.9: results.append(create_det(mock_target_pos[2]['x'], mock_target_pos[2]['y'], 2))
    if random.random() < 0.7: results.append(create_det(mock_target_pos[1]['x'], mock_target_pos[1]['y'], 1))
    if random.random() < 0.6: results.append(create_det(mock_target_pos[3]['x'], mock_target_pos[3]['y'], 3))
    if random.random() < 0.5: results.append(create_det(mock_target_pos[0]['x'], mock_target_pos[0]['y'], 0))
    return results

def post_process_yolo_output(output, input_shape, original_shape, conf_threshold=0.5, nms_threshold=0.4):
    if len(output.shape) == 3: output = output[0]
    if output.shape[0] < output.shape[1]: output = output.T
    boxes, scores, class_scores = output[:, :4], output[:, 4], output[:, 5:]
    mask = scores > conf_threshold
    boxes, scores, class_scores = boxes[mask], scores[mask], class_scores[mask]
    if len(boxes) == 0: return []
    
    class_ids = np.argmax(class_scores, axis=1)
    final_scores = scores * np.max(class_scores, axis=1)
    scale_x, scale_y = original_shape[1]/input_shape[1], original_shape[0]/input_shape[0]
    x_center, y_center = boxes[:, 0]*scale_x, boxes[:, 1]*scale_y
    width, height = boxes[:, 2]*scale_x, boxes[:, 3]*scale_y
    x1, y1 = x_center - width/2, y_center - height/2
    x2, y2 = x_center + width/2, y_center + height/2
    
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), final_scores.tolist(), conf_threshold, nms_threshold)
    return [{'xyxy': [x1[i], y1[i], x2[i], y2[i]], 'cls': int(class_ids[i]), 'conf': float(final_scores[i])} 
            for i in (indices.flatten() if len(indices) > 0 else [])]

# ===== DRONE CONTROL =====
async def arm_and_takeoff(drone, target_alt):
    print("Waiting for drone...")
    async for state in drone.telemetry.health():
        if all([state.is_gyrometer_calibration_ok, state.is_accelerometer_calibration_ok, 
                state.is_magnetometer_calibration_ok, state.is_home_position_ok]):
            break
        await asyncio.sleep(1)
    
    await asyncio.sleep(2)
    try:
        await drone.action.hold()
        await asyncio.sleep(1)
    except: pass
    
    try:
        await drone.action.arm()
        await asyncio.sleep(1)
    except Exception as e:
        print(f"Arming failed: {e}")
        return False
    
    await drone.offboard.set_attitude_rate(AttitudeRate(0, 0, 0, TAKEOFF_THRUST_VALUE))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard failed: {e}")
        return False
    
    start = time.time()
    while True:
        async for pos in drone.telemetry.position():
            alt = pos.relative_altitude_m
            break
        await drone.offboard.set_attitude_rate(AttitudeRate(0, 0, 0, TAKEOFF_THRUST_VALUE))
        if alt >= target_alt * 0.95 or alt >= ALTITUDE_CAP_M: break
        if time.time() - start > 15: return False
        await asyncio.sleep(0.1)
    
    await drone.offboard.set_attitude_rate(AttitudeRate(0, 0, 0, HOVER_THRUST_VALUE))
    return True

async def send_attitude_rate_commands(drone, r, p, y, t):
    await drone.offboard.set_attitude_rate(AttitudeRate(
        np.deg2rad(np.clip(r, -PITCH_LIMIT_ALIGN, PITCH_LIMIT_ALIGN)),
        np.deg2rad(-np.clip(p, -PITCH_LIMIT_ALIGN, PITCH_LIMIT_ALIGN)),
        np.deg2rad(np.clip(y, -YAW_LIMIT, YAW_LIMIT)), t))

async def send_velocity_commands(drone, n, e, d, yaw_deg):
    await drone.offboard.set_velocity_ned(VelocityNedYaw(
        np.clip(n, MAX_BACKWARD_VELOCITY, MAX_FORWARD_VELOCITY),
        np.clip(e, -MAX_FORWARD_VELOCITY, MAX_FORWARD_VELOCITY),
        np.clip(d, -1.0, 1.0), yaw_deg))

# ===== MAIN TRACKING LOOP =====
async def tracking_loop():
    drone = System()
    await drone.connect(system_address=CONNECTION_STRING)
    
    async for state in drone.telemetry.health():
        if all([state.is_gyrometer_calibration_ok, state.is_accelerometer_calibration_ok,
                state.is_magnetometer_calibration_ok, state.is_home_position_ok]):
            break
        await asyncio.sleep(1)
    
    while True:
        async for fm in drone.telemetry.flight_mode():
            if fm == FlightMode.OFFBOARD: break
            await asyncio.sleep(1)
        if fm == FlightMode.OFFBOARD: break
    
    if not await arm_and_takeoff(drone, 2.0): return
    
    yaw_pid = PID(KP_YAW, KI_YAW, KD_YAW)
    pitch_pid = PID(KP_PITCH_ALIGN, KI_PITCH_ALIGN, KD_PITCH_ALIGN)
    follow_pid = PID(KP_FOLLOW, KI_FOLLOW, KD_FOLLOW)
    
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER) if USE_GSTREAMER_CAMERA else cv2.VideoCapture(0)
    if not cap.isOpened(): return
    
    ret, frame = cap.read()
    if not ret: return
    
    h, w, _ = frame.shape
    cx, cy = w/2, h/2
    initialize_mock_targets(w, h)
    
    trt_engine, use_trt = None, USE_REAL_YOLO and TRT_AVAILABLE
    if use_trt:
        try:
            trt_engine = TensorRTInference(REAL_YOLO_MODEL_PATH)
            use_trt = trt_engine.load_engine()
        except:
            use_trt = False
    
    yaw_kf = KalmanFilter(0.1, Q_KF, R_KF, cx)
    pitch_kf = KalmanFilter(0.1, Q_KF, R_KF, cy)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(VIDEO_FOLDER, f"flight_data_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    
    csv_file = open(os.path.join(folder, f"log_{timestamp}.csv"), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Time_s', 'FlightMode', 'Raw_Yaw_Err_px', 'KF_Yaw_Est_px', 'Yaw_Cmd_deg_s', 'Yaw_I_term',
                         'Raw_Pitch_Err_px', 'KF_Pitch_Est_px', 'Pitch_Cmd_deg_s', 'Pitch_I_term',
                         'Class_ID', 'Alt_m', 'Box_Ratio', 'Follow_Vel_mps', 'Follow_I'])
    
    video = None
    if RECORD_VIDEO:
        video = cv2.VideoWriter(os.path.join(folder, f"video_{timestamp}.mp4"), VIDEO_CODEC, VIDEO_FPS, (w, h))
    
    offboard_active = True
    last_time = time.time()
    
    while True:
        t = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        dt = t - last_time
        last_time = t
        
        yaw_kf.dt = pitch_kf.dt = dt
        yaw_kf.predict()
        pitch_kf.predict()
        
        async for fm in drone.telemetry.flight_mode():
            mode = fm
            break
        async for pos in drone.telemetry.position():
            alt = pos.relative_altitude_m
            break
        
        log_row = [t, mode.name]
        
        if mode == FlightMode.OFFBOARD:
            if not offboard_active:
                try:
                    await drone.offboard.start()
                    offboard_active = True
                except: pass
            
            dets = mock_yolo_inference(frame)
            if use_trt and trt_engine:
                try:
                    ih, iw = trt_engine.get_input_shape()[2:]
                    img = cv2.resize(frame, (iw, ih))
                    img = np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0, (2,0,1))
                    dets = post_process_yolo_output(trt_engine.infer(np.expand_dims(img, 0))[0], (ih,iw), (h,w))
                except:
                    pass
            
            sel = None
            max_pri = -1
            for d in dets:
                pri = CLASS_PRIORITIES.get(d['cls'], -1)
                if pri > max_pri:
                    max_pri, sel = pri, d
            
            raw_yaw, kf_yaw, yaw_cmd, yaw_i = 0, yaw_kf.get_estimated_state()[0], 0, yaw_pid.I_term
            raw_pitch, kf_pitch, pitch_cmd, pitch_i = 0, pitch_kf.get_estimated_state()[0], 0, pitch_pid.I_term
            cid, ratio, fvel, fi = -1, 0, 0, follow_pid.I_term
            
            if sel:
                x1, y1, x2, y2 = sel['xyxy']
                bx, by = (x1+x2)/2, (y1+y2)/2
                ratio = (y2-y1)/h
                
                yaw_kf.update(np.array([[bx]]))
                pitch_kf.update(np.array([[by]]))
                ex, _ = yaw_kf.get_estimated_state()
                ey, _ = pitch_kf.get_estimated_state()
                
                yerr, perr = ex-cx, ey-cy
                yaw_cmd = yaw_pid.update(yerr, dt) if abs(yerr) > DEAD_ZONE_YAW else (yaw_pid.reset() or 0)
                pitch_cmd = pitch_pid.update(perr, dt) if abs(perr) > DEAD_ZONE_PITCH_ALIGN else (pitch_pid.reset() or 0)
                
                if ENABLE_FOLLOWING:
                    serr = TARGET_BOX_SIZE_RATIO - ratio
                    fvel = np.clip(follow_pid.update(serr, dt), MAX_BACKWARD_VELOCITY, MAX_FORWARD_VELOCITY) if abs(serr) > DEAD_ZONE_FOLLOW else (follow_pid.reset() or 0)
                    fi = follow_pid.I_term
                    async for att in drone.telemetry.attitude_euler():
                        yaw_deg = att.yaw_deg
                        break
                    await send_velocity_commands(drone, fvel, 0, 0, yaw_deg)
                else:
                    await send_attitude_rate_commands(drone, 0, pitch_cmd, yaw_cmd, HOVER_THRUST_VALUE)
                
                raw_yaw, kf_yaw, yaw_i = bx-cx, ex, yaw_pid.I_term
                raw_pitch, kf_pitch, pitch_i = by-cy, ey, pitch_pid.I_term
                cid = sel['cls']
                
                for d in dets:
                    dx1, dy1, dx2, dy2 = map(int, d['xyxy'])
                    cv2.rectangle(frame, (dx1,dy1), (dx2,dy2), (0,255,0) if d==sel else (0,100,255), 2)
                cv2.circle(frame, (int(bx), int(by)), 5, (0,0,255), -1)
                cv2.circle(frame, (int(ex), int(ey)), 8, (255,0,255), -1)
            else:
                ex, _ = yaw_kf.get_estimated_state()
                ey, _ = pitch_kf.get_estimated_state()
                yerr, perr = ex-cx, ey-cy
                yaw_cmd = yaw_pid.update(yerr, dt) if abs(yerr) > DEAD_ZONE_YAW else (yaw_pid.reset() or 0)
                pitch_cmd = pitch_pid.update(perr, dt) if abs(perr) > DEAD_ZONE_PITCH_ALIGN else (pitch_pid.reset() or 0)
                
                if ENABLE_FOLLOWING:
                    async for att in drone.telemetry.attitude_euler():
                        yaw_deg = att.yaw_deg
                        break
                    await send_velocity_commands(drone, 0, 0, 0, yaw_deg)
                else:
                    await send_attitude_rate_commands(drone, 0, pitch_cmd, yaw_cmd, HOVER_THRUST_VALUE)
                
                kf_yaw, kf_pitch = ex, ey
                cv2.circle(frame, (int(ex), int(ey)), 8, (255,165,0), -1)
            
            cv2.line(frame, (int(cx), 0), (int(cx), h), (255,0,0), 2)
            cv2.line(frame, (0, int(cy)), (w, int(cy)), (255,0,0), 2)
            cv2.putText(frame, f"Tracking: ON | ID: {cid}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Yaw: {yerr:.1f}px | {yaw_cmd:.1f}deg/s", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Pitch: {perr:.1f}px | {pitch_cmd:.1f}deg/s", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            if ENABLE_FOLLOWING:
                cv2.putText(frame, f"Follow: {fvel:.2f}m/s | Ratio: {ratio:.2f}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(frame, f"Alt: {alt:.1f}m | TRT: {'ON' if use_trt else 'OFF'}", (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        else:
            if offboard_active:
                try:
                    await drone.offboard.stop()
                    offboard_active = False
                except: pass
            await send_attitude_rate_commands(drone, 0, 0, 0, HOVER_THRUST_VALUE)
            yaw_pid.reset()
            pitch_pid.reset()
            follow_pid.reset()
            yaw_kf.reset(cx)
            pitch_kf.reset(cy)
            raw_yaw = kf_yaw = yaw_cmd = yaw_i = 0
            raw_pitch = kf_pitch = pitch_cmd = pitch_i = 0
            cid = ratio = fvel = fi = 0
            cv2.putText(frame, f"Tracking: OFF ({mode.name})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        log_row.extend([raw_yaw, kf_yaw, yaw_cmd, yaw_i, raw_pitch, kf_pitch, pitch_cmd, pitch_i, cid, alt, ratio, fvel, fi])
        csv_writer.writerow(log_row)
        if video: video.write(frame)
        cv2.imshow("UAV Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    if offboard_active:
        try: await drone.offboard.stop()
        except: pass
    await drone.action.land()
    await asyncio.sleep(5)
    await drone.action.disarm()
    
    cap.release()
    if video: video.release()
    csv_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(tracking_loop())