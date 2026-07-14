# Adaptive Semantic Priority Selection (ASPS)
# Mission-defined hierarchy with runtime refinement.
# P_final = ALPHA*P_base + BETA*Conf + GAMMA*Distance + DELTA*Velocity

import math
import time

# --- Configuration ---
ALPHA = 0.60       # Base priority weight (mission hierarchy)
BETA  = 0.20       # Confidence weight
GAMMA = 0.15       # Distance weight
DELTA = 0.05       # Velocity weight

SWITCH_THRESHOLD = 0.15   # Hysteresis: only switch targets when new > current + threshold
USE_ASPS = True           # False = pure CLASS_PRIORITIES (ablation baseline)


class VelocityTracker:
    """
    Tracks per-class bounding-box velocity with exponential smoothing.
    Keyed by class_id: stores (cx, cy, timestamp, filtered_velocity).
    """
    def __init__(self, smoothing=0.8):
        self.smoothing = smoothing
        self._data = {}

    def update(self, detections, frame_time):
        for det in detections:
            cls_id = det['cls']
            x1, y1, x2, y2 = det['xyxy']
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            prev = self._data.get(cls_id)
            if prev is not None:
                dt = max(frame_time - prev['t'], 0.001)
                dx = cx - prev['cx']
                dy = cy - prev['cy']
                raw_vel = math.sqrt(dx * dx + dy * dy) / dt
                # Exponential smoothing
                filtered = (self.smoothing * prev['v'] +
                            (1.0 - self.smoothing) * raw_vel)
            else:
                filtered = 0.0

            self._data[cls_id] = {'cx': cx, 'cy': cy, 't': frame_time, 'v': filtered}

    def get_velocity_score(self, class_id):
        """Normalized [0, 1] velocity score. Clips at 500 px/s ≈ full motion."""
        entry = self._data.get(class_id)
        if entry is None:
            return 0.0
        return min(entry['v'] / 500.0, 1.0)

    def reset(self):
        self._data.clear()


def compute_distance_score(distance_m):
    """
    Distance component: nearby targets score higher.
    1m -> 1.00,  2m -> 0.50,  5m -> 0.20,  10m -> 0.10
    Returns 0.0 if distance is None or unavailable.
    """
    if distance_m is None or distance_m <= 0:
        return 0.0
    return min(1.0, 1.0 / max(distance_m, 1.0))


def compute_priority(class_id, confidence, distance_m, velocity_score, class_priorities):
    """
    Compute final ASPS priority score with component breakdown.

    Returns: (priority_score: float, components: dict)
    """
    max_p = max(class_priorities.values())
    base_norm = class_priorities.get(class_id, 0) / max_p   # [0.25 .. 1.00]

    conf = max(0.0, min(confidence, 1.0))
    dist = compute_distance_score(distance_m)
    vel  = max(0.0, min(velocity_score, 1.0))

    total = (ALPHA * base_norm +
             BETA  * conf +
             GAMMA * dist +
             DELTA * vel)

    components = {
        'base_priority': round(ALPHA * base_norm, 3),
        'confidence':    round(BETA  * conf,      3),
        'distance':      round(GAMMA * dist,      3),
        'velocity':      round(DELTA * vel,       3),
    }
    return round(total, 3), components


class ASPSPrioritySelector:
    """
    Target selector using ASPS scoring with hysteresis.

    Usage:
        selector = ASPSPrioritySelector()
        best_det, components = selector.select(detections, time.time(),
                                                CLASS_PRIORITIES, distance_fn)
    """
    def __init__(self):
        self.vt = VelocityTracker()
        self.current_class = None
        self.current_score = 0.0

    def select(self, detections, frame_time, class_priorities, distance_fn=None):
        """
        detections:   list of {'cls': int, 'conf': float, 'xyxy': [x1,y1,x2,y2]}
        frame_time:   time.time() for velocity computation
        class_priorities: dict {class_id: base_priority}
        distance_fn:  callable(detection) -> float or None (per-target distance)

        Returns:
            (best_detection or None, components dict or None)
        """
        if not USE_ASPS or not detections:
            # Legacy fallback: pure CLASS_PRIORITIES max
            best = max(detections, key=lambda d: class_priorities.get(d['cls'], 0)) if detections else None
            return best, None

        # Update velocity tracker with current frame
        self.vt.update(detections, frame_time)

        # Score every detection
        scored = []
        for det in detections:
            dist = distance_fn(det) if distance_fn else None
            vel  = self.vt.get_velocity_score(det['cls'])
            score, comps = compute_priority(
                det['cls'], det.get('conf', 0.5), dist, vel, class_priorities
            )
            scored.append((det, score, comps))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        best_det, best_score, best_comps = scored[0]

        # Hysteresis: only switch if new > current + threshold
        if (self.current_class is not None
                and best_det['cls'] != self.current_class
                and best_score < self.current_score + SWITCH_THRESHOLD):
            # Keep current target — find it in scored list
            for det, sc, comp in scored:
                if det['cls'] == self.current_class:
                    best_det, best_score, best_comps = det, sc, comp
                    break

        self.current_class = best_det['cls']
        self.current_score = best_score
        return best_det, best_comps

    def reset(self):
        self.current_class = None
        self.current_score = 0.0
        self.vt.reset()
