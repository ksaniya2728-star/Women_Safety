"""Module 4: interaction-feature analysis for tracked people and pose keypoints.

This module consumes tracking/pose output (for example from Module 3).  It does
not run YOLO and does not make behavioural, criminal, or safety claims.  All
outputs are image-space geometric and temporal *features* for a later consumer.
Run ``python interactive_analyser_M4.py`` for an optional synthetic-data GUI.
"""
from __future__ import annotations
 
import json
import logging
import math
import os
import queue
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from itertools import combinations
from pathlib import Path
import tkinter as tk
from tkinter import BooleanVar, StringVar, Tk, messagebox, ttk
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ------------------------------ Configuration ------------------------------
LOG_LEVEL = logging.INFO
KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                  "left_wrist", "right_wrist", "left_hip", "right_hip",
                  "left_knee", "right_knee", "left_ankle", "right_ankle"]
KEYPOINT_CONFIDENCE_THRESHOLD = 0.40
MIN_PERSON_HEIGHT = 20.0
VERY_CLOSE_THRESHOLD, CLOSE_THRESHOLD, MODERATE_THRESHOLD = 0.85, 1.65, 3.0
NEARBY_FILTER_THRESHOLD = MODERATE_THRESHOLD
MOVING_SPEED_THRESHOLD, FAST_MOVING_SPEED_THRESHOLD = 8.0, 45.0
APPROACH_HISTORY_LENGTH, APPROACH_MIN_DURATION = 10, 0.60
APPROACH_SLOPE_THRESHOLD = 0.08
FOLLOWING_HISTORY_LENGTH, FOLLOWING_SCORE_THRESHOLD = 12, 0.65
FOLLOWING_MIN_DURATION = 1.0
FOLLOWING_DIRECTION_WEIGHT = 0.28
FOLLOWING_VELOCITY_WEIGHT = 0.20
FOLLOWING_TRAJECTORY_WEIGHT = 0.25
FOLLOWING_DISTANCE_WEIGHT = 0.17
FOLLOWING_DURATION_WEIGHT = 0.10
FACING_SCORE_THRESHOLD = 0.60
HAND_PROXIMITY_THRESHOLD = 0.80
PAIR_TIMEOUT_SECONDS = 3.0
MIN_STATE_DURATION = 0.35
HISTORY_WINDOW = 20
INTERACTION_PROXIMITY_WEIGHT = 0.24
INTERACTION_APPROACH_WEIGHT = 0.18
INTERACTION_FOLLOWING_WEIGHT = 0.18
INTERACTION_FACING_WEIGHT = 0.14
INTERACTION_HAND_WEIGHT = 0.16
INTERACTION_TEMPORAL_WEIGHT = 0.10
GUI_UPDATE_INTERVAL_MS = 100
SHOW_INTERACTION_LINES = True
SHOW_PROXIMITY_LABELS = True
SHOW_APPROACH_LABELS = True
SHOW_FACING_LABELS = True
SHOW_HAND_LABELS = True

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
Point = Tuple[float, float]
Person = Mapping[str, Any]

# Integrated Module-3 camera/model configuration (models are loaded once).
DEFAULT_CAMERA_URL = "http://192.168.1.10:8080/video"
MODEL_PATH = os.path.join("models", "yolo11n.pt")
POSE_MODEL_PATH = os.path.join("models", "yolo11n-pose.pt")
WEAPON_MODEL_PATH = os.path.join("models", "weapon.pt")
CONFIDENCE_THRESHOLD = 0.50
WEAPON_CONFIDENCE_THRESHOLD = 0.45
INFERENCE_WIDTH, INFERENCE_HEIGHT = 640, 640
DETECTION_INTERVAL = 1
PERSON_CLASS_ID = 0
CAPTURE_BUFFER_SIZE = 1
RECONNECT_DELAY, PREVIEW_UPDATE_MS = 3, 30
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
            (13, 15), (12, 14), (14, 16)]


def _center(person: Person) -> Point:
    """Return supplied center or calculate it from a validated bounding box."""
    if person.get("center") is not None:
        x, y = person["center"]
        return float(x), float(y)
    x1, y1, x2, y2 = person["bbox"]
    return (float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2


def _height(person: Person) -> float:
    x1, y1, x2, y2 = person["bbox"]
    return max(abs(float(y2) - float(y1)), MIN_PERSON_HEIGHT)


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit(v: Point) -> Optional[Point]:
    length = math.hypot(*v)
    return (v[0] / length, v[1] / length) if length else None


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _keypoint(person: Person, name: str) -> Optional[Point]:
    """Return a visible keypoint. Missing/low-confidence joints stay unavailable."""
    kp = person.get("keypoints", {}).get(name)
    if not kp or float(kp.get("confidence", 0.0)) < KEYPOINT_CONFIDENCE_THRESHOLD:
        return None
    if kp.get("x") is None or kp.get("y") is None:
        return None
    return float(kp["x"]), float(kp["y"])


class PairDetector:
    """Generates unique, ID-normalized candidate person pairs."""
    def candidate_pairs(self, people: Sequence[Person]) -> Iterable[Tuple[Person, Person]]:
        valid = sorted((p for p in people if p.get("person_id") is not None and p.get("bbox")),
                       key=lambda p: int(p["person_id"]))
        return combinations(valid, 2)

    @staticmethod
    def key(a: Person, b: Person) -> Tuple[int, int]:
        return tuple(sorted((int(a["person_id"]), int(b["person_id"]))))


class ProximityAnalyzer:
    """Calculates relative image-space proximity; it is not physical distance."""
    def calculate(self, a: Person, b: Person) -> Dict[str, Any]:
        pixel = _distance(_center(a), _center(b))
        normalized = pixel / ((_height(a) + _height(b)) / 2.0)
        if normalized <= VERY_CLOSE_THRESHOLD:
            level = "VERY_CLOSE"
        elif normalized <= CLOSE_THRESHOLD:
            level = "CLOSE"
        elif normalized <= MODERATE_THRESHOLD:
            level = "MODERATE"
        else:
            level = "FAR"
        return {"pixel_distance": pixel, "normalized_distance": normalized,
                "proximity_level": level}


class DistanceAnalyzer:
    """Adds previous-distance and delta data to a proximity observation."""
    def calculate(self, a: Person, b: Person, previous: Optional[float]) -> Dict[str, Any]:
        data = ProximityAnalyzer().calculate(a, b)
        data["previous_distance"] = previous
        data["distance_delta"] = None if previous is None else data["normalized_distance"] - previous
        return data


class MovementAnalyzer:
    """Maintains timestamped image-space trajectories and movement features."""
    def __init__(self) -> None:
        self.positions: Dict[int, Deque[Tuple[float, Point]]] = defaultdict(
            lambda: deque(maxlen=FOLLOWING_HISTORY_LENGTH))

    def update(self, people: Sequence[Person], timestamp: float) -> Dict[int, Dict[str, Any]]:
        results: Dict[int, Dict[str, Any]] = {}
        for person in people:
            pid, pos = int(person["person_id"]), _center(person)
            history = self.positions[pid]
            vx = vy = speed = 0.0
            if history:
                old_t, old_pos = history[-1]
                dt = timestamp - old_t
                if dt > 0:
                    vx, vy = (pos[0] - old_pos[0]) / dt, (pos[1] - old_pos[1]) / dt
                    speed = math.hypot(vx, vy)
            history.append((timestamp, pos))
            state = "FAST_MOVING" if speed >= FAST_MOVING_SPEED_THRESHOLD else (
                "MOVING" if speed >= MOVING_SPEED_THRESHOLD else "STATIONARY")
            results[pid] = {"velocity_x": vx, "velocity_y": vy, "speed": speed,
                            "movement_direction": _unit((vx, vy)), "movement_state": state}
        return results

    def trajectory(self, person_id: int) -> Sequence[Tuple[float, Point]]:
        return self.positions.get(person_id, ())


class ApproachAnalyzer:
    """Uses a multi-frame normalized-distance trend, never a single frame."""
    def calculate(self, state: "InteractionState", movements: Mapping[int, Dict[str, Any]]) -> Dict[str, Any]:
        history = state.distance_history
        if len(history) < 3 or history[-1][0] - history[0][0] < APPROACH_MIN_DURATION:
            return {"approaching": False, "separating": False, "distance_trend": "UNKNOWN",
                    "approach_score": 0.0, "target_moving_away": False,
                    "approacher_id": None, "target_id": None}
        ts, ds = zip(*history)
        mean_t, mean_d = statistics.mean(ts), statistics.mean(ds)
        denom = sum((t - mean_t) ** 2 for t in ts)
        slope = sum((t - mean_t) * (d - mean_d) for t, d in zip(ts, ds)) / denom if denom else 0.0
        consistency = sum(1 for x, y in zip(ds, ds[1:]) if y < x) / (len(ds) - 1)
        approaching, separating = slope < -APPROACH_SLOPE_THRESHOLD and consistency >= .65, slope > APPROACH_SLOPE_THRESHOLD and consistency <= .35
        score = min(1.0, abs(slope) / (APPROACH_SLOPE_THRESHOLD * 4)) * (consistency if slope < 0 else 0.0)
        # Infer the likely approacher only from relative movement along their connecting axis.
        va, vb = movements.get(state.person_a, {}), movements.get(state.person_b, {})
        direction = _unit((state.last_center_b[0] - state.last_center_a[0], state.last_center_b[1] - state.last_center_a[1])) if state.last_center_a and state.last_center_b else None
        approacher = target = None
        target_away = False
        if approaching and direction:
            a_toward = _dot((va.get("velocity_x", 0), va.get("velocity_y", 0)), direction)
            b_away = _dot((vb.get("velocity_x", 0), vb.get("velocity_y", 0)), direction)
            if a_toward >= -b_away:
                approacher, target, target_away = state.person_a, state.person_b, b_away > MOVING_SPEED_THRESHOLD / 2
            else:
                approacher, target, target_away = state.person_b, state.person_a, -a_toward > MOVING_SPEED_THRESHOLD / 2
        return {"approaching": approaching, "separating": separating,
                "distance_trend": "DECREASING" if approaching else "INCREASING" if separating else "STABLE",
                "approach_score": score, "target_moving_away": target_away,
                "approacher_id": approacher, "target_id": target}


class FollowingAnalyzer:
    """Heuristic trajectory-similarity feature; it is not a definitive classification."""
    def calculate(self, state: "InteractionState", movement: MovementAnalyzer,
                  motions: Mapping[int, Dict[str, Any]], timestamp: float) -> Dict[str, Any]:
        a, b = motions.get(state.person_a, {}), motions.get(state.person_b, {})
        va, vb = (a.get("velocity_x", 0.0), a.get("velocity_y", 0.0)), (b.get("velocity_x", 0.0), b.get("velocity_y", 0.0))
        ua, ub = _unit(va), _unit(vb)
        direction_score = max(0.0, _dot(ua, ub)) if ua and ub else 0.0
        sa, sb = a.get("speed", 0.0), b.get("speed", 0.0)
        velocity_score = 1.0 - min(1.0, abs(sa - sb) / max(sa, sb, 1.0))
        ta, tb = movement.trajectory(state.person_a), movement.trajectory(state.person_b)
        count = min(len(ta), len(tb), FOLLOWING_HISTORY_LENGTH)
        trajectory_score = 0.0
        if count >= 3:
            da = _unit((ta[-1][1][0] - ta[-count][1][0], ta[-1][1][1] - ta[-count][1][1]))
            db = _unit((tb[-1][1][0] - tb[-count][1][0], tb[-1][1][1] - tb[-count][1][1]))
            trajectory_score = max(0.0, _dot(da, db)) if da and db else 0.0
        distances = [d for _, d in state.distance_history]
        stability = 1.0 - min(1.0, (statistics.pstdev(distances) if len(distances) > 1 else 1.0) / .75)
        duration_score = min(1.0, max(0.0, timestamp - state.first_seen) / FOLLOWING_MIN_DURATION)
        score = (FOLLOWING_DIRECTION_WEIGHT * direction_score + FOLLOWING_VELOCITY_WEIGHT * velocity_score +
                 FOLLOWING_TRAJECTORY_WEIGHT * trajectory_score + FOLLOWING_DISTANCE_WEIGHT * stability +
                 FOLLOWING_DURATION_WEIGHT * duration_score)
        return {"following_score": score, "following": score >= FOLLOWING_SCORE_THRESHOLD and count >= 3}


class FacingAnalyzer:
    """Rough shoulder-based orientation; unavailable shoulder data yields UNKNOWN."""
    @staticmethod
    def _orientation_alignment(person: Person, other: Point) -> Optional[float]:
        left, right = _keypoint(person, "left_shoulder"), _keypoint(person, "right_shoulder")
        if not left or not right:
            return None
        axis = _unit((right[0] - left[0], right[1] - left[1]))
        if not axis:
            return None
        midpoint = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        # A monocular shoulder axis cannot reliably resolve front versus back. Its
        # perpendicular still provides a rough body-orientation alignment feature.
        toward = _unit((other[0] - midpoint[0], other[1] - midpoint[1]))
        return abs(_dot((-axis[1], axis[0]), toward)) if toward else None

    def calculate(self, a: Person, b: Person) -> Dict[str, Any]:
        ca, cb = _center(a), _center(b)
        aa, ab = self._orientation_alignment(a, cb), self._orientation_alignment(b, ca)
        if aa is None or ab is None:
            return {"facing_status": "UNKNOWN", "facing_each_other": None, "facing_score": None}
        score = (aa + ab) / 2.0
        return {"facing_status": "FACING" if score >= FACING_SCORE_THRESHOLD else "NOT_FACING",
                "facing_each_other": score >= FACING_SCORE_THRESHOLD, "facing_score": score}


class HandInteractionAnalyzer:
    """Reports wrist-to-other-body proximity as a neutral geometric feature."""
    @staticmethod
    def _body_points(person: Person) -> List[Point]:
        names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        points = [p for name in names if (p := _keypoint(person, name))]
        if len(points) >= 2:
            points.append((sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)))
        return points

    def _side(self, actor: Person, target: Person) -> Tuple[bool, Dict[str, Optional[float]]]:
        body, scale = self._body_points(target), _height(target)
        values: Dict[str, Optional[float]] = {}
        for side in ("left", "right"):
            wrist = _keypoint(actor, f"{side}_wrist")
            values[f"{side}_wrist_to_body"] = min((_distance(wrist, p) / scale for p in body), default=None) if wrist else None
        available = [v for v in values.values() if v is not None]
        return bool(available and min(available) <= HAND_PROXIMITY_THRESHOLD), values

    def calculate(self, a: Person, b: Person) -> Dict[str, Any]:
        a_near, av = self._side(a, b)
        b_near, bv = self._side(b, a)
        vals = [v for v in (*av.values(), *bv.values()) if v is not None]
        return {"a_hand_near_b_body": a_near, "b_hand_near_a_body": b_near,
                "a_left_wrist_to_b_torso": av["left_wrist_to_body"], "a_right_wrist_to_b_torso": av["right_wrist_to_body"],
                "b_left_wrist_to_a_torso": bv["left_wrist_to_body"], "b_right_wrist_to_a_torso": bv["right_wrist_to_body"],
                "hand_near_body": a_near or b_near,
                "hand_near_body_score": 0.0 if not vals else max(0.0, 1 - min(vals) / HAND_PROXIMITY_THRESHOLD)}


class TemporalAnalyzer:
    """Applies short persistence gates and maintains duration counters."""
    def update(self, state: "InteractionState", timestamp: float, raw: Mapping[str, Any]) -> Dict[str, Any]:
        dt = max(0.0, timestamp - state.last_seen) if state.last_seen else 0.0
        for name in ("approaching", "separating", "following", "facing_each_other", "hand_near_body"):
            value = bool(raw.get(name))
            history = state.feature_history[name]
            history.append((timestamp, value))
            # A state requires persistence; false values do not turn unknown into a claim.
            sustained = value and len(history) >= 2 and timestamp - history[0][0] >= MIN_STATE_DURATION and sum(v for _, v in history) / len(history) >= .60
            raw[name] = sustained
        if state.proximity_level in ("VERY_CLOSE", "CLOSE"):
            state.close_duration += dt
        if raw["approaching"]: state.approach_duration += dt
        if raw["following"]: state.following_duration += dt
        if raw.get("facing_each_other"): state.facing_duration += dt
        if raw["hand_near_body"]: state.hand_proximity_duration += dt
        return dict(raw)


@dataclass
class InteractionState:
    """Bounded temporal state for one normalized pair of tracking IDs."""
    person_a: int
    person_b: int
    first_seen: float
    last_seen: float = 0.0
    current_distance: Optional[float] = None
    previous_distance: Optional[float] = None
    normalized_distance: Optional[float] = None
    proximity_level: str = "FAR"
    close_duration: float = 0.0
    approach_duration: float = 0.0
    following_duration: float = 0.0
    facing_duration: float = 0.0
    hand_proximity_duration: float = 0.0
    distance_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=APPROACH_HISTORY_LENGTH))
    feature_history: Dict[str, Deque[Tuple[float, bool]]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=HISTORY_WINDOW)))
    last_center_a: Optional[Point] = None
    last_center_b: Optional[Point] = None


class InteractionStateManager:
    """Creates, updates, and time-cleans pair state to bound memory use."""
    def __init__(self) -> None:
        self.states: Dict[Tuple[int, int], InteractionState] = {}

    def get(self, key: Tuple[int, int], timestamp: float) -> InteractionState:
        if key not in self.states:
            self.states[key] = InteractionState(*key, first_seen=timestamp)
        return self.states[key]

    def cleanup(self, timestamp: float) -> None:
        stale = [key for key, state in self.states.items() if timestamp - state.last_seen > PAIR_TIMEOUT_SECONDS]
        for key in stale:
            del self.states[key]


class InteractionManager:
    """Coordinates proximity-first interaction feature extraction and optional worker use."""
    def __init__(self) -> None:
        self.pairs, self.proximity, self.distance = PairDetector(), ProximityAnalyzer(), DistanceAnalyzer()
        self.movement, self.approach, self.following = MovementAnalyzer(), ApproachAnalyzer(), FollowingAnalyzer()
        self.facing, self.hand, self.temporal, self.states = FacingAnalyzer(), HandInteractionAnalyzer(), TemporalAnalyzer(), InteractionStateManager()
        self.enabled = {name: True for name in ("pair", "proximity", "distance", "movement", "approach", "following", "facing", "hand", "temporal")}
        self.stats = {"processing_fps": 0.0, "average_analysis_ms": 0.0}
        self._durations: Deque[float] = deque(maxlen=60)
        self.input_queue: "queue.Queue[Tuple[Sequence[Person], float]]" = queue.Queue(maxsize=1)
        self._latest: Optional[Dict[str, Any]] = None
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def process(self, people: Sequence[Person], timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Process one Module-3-style batch synchronously and return feature results."""
        started, timestamp = time.perf_counter(), time.time() if timestamp is None else timestamp
        people = list(people)
        motions = self.movement.update(people, timestamp) if self.enabled["movement"] else {}
        candidates = list(self.pairs.candidate_pairs(people)) if self.enabled["pair"] else []
        output: List[Dict[str, Any]] = []
        nearby = 0
        for a, b in candidates:
            cheap = self.proximity.calculate(a, b)
            if cheap["normalized_distance"] > NEARBY_FILTER_THRESHOLD:
                continue  # Performance gate: detailed analysis only starts here for nearby pairs.
            nearby += 1
            key, state = self.pairs.key(a, b), self.states.get(self.pairs.key(a, b), timestamp)
            distance = self.distance.calculate(a, b, state.current_distance) if self.enabled["distance"] else cheap
            state.previous_distance, state.current_distance = state.current_distance, distance["normalized_distance"]
            state.normalized_distance, state.proximity_level = distance["normalized_distance"], distance["proximity_level"]
            state.distance_history.append((timestamp, state.normalized_distance))
            state.last_center_a, state.last_center_b = _center(a), _center(b)
            raw: Dict[str, Any] = {**distance, "person_a": key[0], "person_b": key[1], **motions.get(key[0], {}),
                                   "a_speed": motions.get(key[0], {}).get("speed", 0.0), "b_speed": motions.get(key[1], {}).get("speed", 0.0)}
            raw.update(self.approach.calculate(state, motions) if self.enabled["approach"] else {})
            raw.update(self.following.calculate(state, self.movement, motions, timestamp) if self.enabled["following"] else {"following": False, "following_score": 0.0})
            raw.update(self.facing.calculate(a, b) if self.enabled["facing"] else {"facing_each_other": None, "facing_score": None, "facing_status": "DISABLED"})
            raw.update(self.hand.calculate(a, b) if self.enabled["hand"] else {"hand_near_body": False, "a_hand_near_b_body": False, "b_hand_near_a_body": False, "hand_near_body_score": 0.0})
            raw = self.temporal.update(state, timestamp, raw) if self.enabled["temporal"] else raw
            state.last_seen = timestamp
            persistence = min(1.0, (timestamp - state.first_seen) / 3.0)
            proximity_score = max(0.0, 1 - state.normalized_distance / MODERATE_THRESHOLD)
            raw["interaction_score"] = min(1.0, INTERACTION_PROXIMITY_WEIGHT * proximity_score + INTERACTION_APPROACH_WEIGHT * raw.get("approach_score", 0) + INTERACTION_FOLLOWING_WEIGHT * raw.get("following_score", 0) + INTERACTION_FACING_WEIGHT * (raw.get("facing_score") or 0) + INTERACTION_HAND_WEIGHT * raw.get("hand_near_body_score", 0) + INTERACTION_TEMPORAL_WEIGHT * persistence)
            raw.update({"close_duration": state.close_duration, "approach_duration": state.approach_duration,
                        "following_duration": state.following_duration, "facing_duration": state.facing_duration,
                        "hand_proximity_duration": state.hand_proximity_duration})
            output.append(raw)
        self.states.cleanup(timestamp)
        elapsed = time.perf_counter() - started
        self._durations.append(elapsed)
        self.stats = {"processing_fps": 1 / statistics.mean(self._durations) if self._durations else 0.0,
                      "average_analysis_ms": statistics.mean(self._durations) * 1000}
        result = {"timestamp": timestamp, "tracked_persons": len(people), "candidate_pairs": len(candidates),
                  "nearby_pairs": nearby, "active_interactions": len(output), "active_pair_states": len(self.states.states),
                  "pairs": output, "performance": self.stats.copy()}
        with self._lock: self._latest = result
        return result

    def submit_latest(self, people: Sequence[Person], timestamp: Optional[float] = None) -> None:
        """Offer a batch to a size-one queue; stale work is discarded, never backlogged."""
        item = (list(people), time.time() if timestamp is None else timestamp)
        try: self.input_queue.put_nowait(item)
        except queue.Full:
            try: self.input_queue.get_nowait()
            except queue.Empty: pass
            self.input_queue.put_nowait(item)

    def start_worker(self) -> threading.Thread:
        if self._worker_thread and self._worker_thread.is_alive():
            return self._worker_thread
        self._stop.clear()
        def work() -> None:
            while not self._stop.is_set():
                try: people, ts = self.input_queue.get(timeout=.2)
                except queue.Empty: continue
                self.process(people, ts)
        thread = threading.Thread(target=work, name="interaction-analysis", daemon=True)
        thread.start(); self._worker_thread = thread
        return thread

    def latest_result(self) -> Optional[Dict[str, Any]]:
        with self._lock: return self._latest

    def stop_worker(self) -> None:
        self._stop.set()
        if self._worker_thread and self._worker_thread.is_alive() and self._worker_thread is not threading.current_thread():
            self._worker_thread.join(timeout=2)

    @staticmethod
    def export_json(result: Mapping[str, Any], path: str | Path) -> None:
        """Export structured feature results only; no camera images are written."""
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, default=str)


class InteractionAnalysisGUI:
    """Compact Tkinter view; it updates row values instead of rebuilding widgets."""
    def __init__(self, manager: InteractionManager) -> None:
        self.manager, self.root = manager, Tk()
        self.root.title("Module 4 — Interaction Analysis Features")
        self.root.geometry("1120x670")
        self.vars = {name: BooleanVar(value=True) for name in manager.enabled}
        controls = ttk.LabelFrame(self.root, text="Interaction Analysis — neutral observable features")
        controls.pack(fill="x", padx=8, pady=6)
        labels = {"pair": "Pair Detection", "proximity": "Proximity", "distance": "Distance", "movement": "Movement", "approach": "Approach", "following": "Following", "facing": "Facing", "hand": "Hand Interaction", "temporal": "Temporal Analysis"}
        for name, label in labels.items():
            ttk.Checkbutton(controls, text=label, variable=self.vars[name], command=self._apply_enabled).pack(side="left", padx=4)
        self.stats = StringVar(value="Waiting for interaction data…")
        ttk.Label(self.root, textvariable=self.stats).pack(anchor="w", padx=12)
        self.table = ttk.Treeview(self.root, columns=("A", "B", "Distance", "Proximity", "Approach", "Following", "Facing", "Hand", "Duration", "Score"), show="headings", height=12)
        for col in self.table["columns"]: self.table.heading(col, text=col); self.table.column(col, width=100, anchor="center")
        self.table.pack(fill="both", expand=True, padx=8, pady=6)
        self.table.bind("<<TreeviewSelect>>", self._select)
        self.details = StringVar(value="Select a nearby pair for detailed feature values.")
        ttk.Label(self.root, textvariable=self.details, justify="left", anchor="nw").pack(fill="x", padx=12, pady=6)
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.root.after(GUI_UPDATE_INTERVAL_MS, self._refresh)

    def _apply_enabled(self) -> None:
        for name, var in self.vars.items(): self.manager.enabled[name] = var.get()

    def _refresh(self) -> None:
        result = self.manager.latest_result()
        if result:
            p = result["performance"]
            self.stats.set(f"Tracked Persons: {result['tracked_persons']}   Candidate Pairs: {result['candidate_pairs']}   Nearby Pairs: {result['nearby_pairs']}   Active Pair States: {result['active_pair_states']}   Interaction FPS: {p['processing_fps']:.1f}   Analysis: {p['average_analysis_ms']:.2f} ms")
            current = set()
            for pair in result["pairs"]:
                iid = f"{pair['person_a']}-{pair['person_b']}"; current.add(iid); self.rows[iid] = pair
                values = (pair["person_a"], pair["person_b"], f"{pair['normalized_distance']:.2f}", pair["proximity_level"], "YES" if pair.get("approaching") else "NO", "YES" if pair.get("following") else "NO", pair.get("facing_status", "UNKNOWN"), "YES" if pair.get("hand_near_body") else "NO", f"{pair.get('close_duration', 0):.1f}s", f"{pair['interaction_score']:.2f}")
                if self.table.exists(iid): self.table.item(iid, values=values)
                else: self.table.insert("", "end", iid=iid, values=values)
            for iid in set(self.table.get_children()) - current: self.table.delete(iid); self.rows.pop(iid, None)
        self.root.after(GUI_UPDATE_INTERVAL_MS, self._refresh)

    def _select(self, _: Any) -> None:
        selected = self.table.selection()
        if not selected: return
        p = self.rows[selected[0]]
        self.details.set(f"Pair {p['person_a']} / {p['person_b']}\nPixel Distance: {p['pixel_distance']:.1f} px | Normalized Distance: {p['normalized_distance']:.2f} | Proximity: {p['proximity_level']}\nA speed: {p.get('a_speed', 0):.1f} px/s | B speed: {p.get('b_speed', 0):.1f} px/s | Approaching: {p.get('approaching')} ({p.get('approach_score', 0):.2f}) | Following: {p.get('following')} ({p.get('following_score', 0):.2f})\nFacing: {p.get('facing_status')} ({p.get('facing_score')}) | Hand Proximity Feature: {p.get('hand_near_body')} | Target Moving Away: {p.get('target_moving_away')} | Interaction Score: {p['interaction_score']:.2f}")

    def run(self) -> None: self.root.mainloop()


def _pose(cx: float, cy: float) -> Dict[str, Dict[str, float]]:
    """Minimal visible joints for the optional synthetic demonstration only."""
    return {"left_shoulder": {"x": cx - 25, "y": cy - 50, "confidence": .9}, "right_shoulder": {"x": cx + 25, "y": cy - 50, "confidence": .9}, "left_hip": {"x": cx - 18, "y": cy + 45, "confidence": .9}, "right_hip": {"x": cx + 18, "y": cy + 45, "confidence": .9}, "left_wrist": {"x": cx - 45, "y": cy, "confidence": .9}, "right_wrist": {"x": cx + 45, "y": cy, "confidence": .9}}


def demo_people(frame: int, timestamp: float) -> List[Dict[str, Any]]:
    """Synthetic approach example, intentionally opt-in via --demo."""
    cx1, cx2, cy = 150 + frame * 3, 420 - frame * 2, 250
    return [{"person_id": 1, "bbox": (cx1 - 50, 100, cx1 + 50, 400), "center": (cx1, cy), "confidence": .95, "keypoints": _pose(cx1, cy), "timestamp": timestamp}, {"person_id": 2, "bbox": (cx2 - 50, 110, cx2 + 50, 410), "center": (cx2, cy), "confidence": .94, "keypoints": _pose(cx2, cy), "timestamp": timestamp}]


class RollingFPS:
    """Thread-safe short-window FPS meter."""
    def __init__(self, seconds: float = 2.0) -> None:
        self.seconds, self.times, self.lock = seconds, deque(), threading.Lock()
    def tick(self) -> None:
        now = time.perf_counter()
        with self.lock:
            self.times.append(now)
            while self.times and self.times[0] < now - self.seconds: self.times.popleft()
    def value(self) -> float:
        with self.lock:
            return (len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times) > 1 and self.times[-1] != self.times[0] else 0.0


class TrackingManager:
    """Lightweight centroid tracker used by the one integrated pipeline."""
    def __init__(self, max_distance: float = 110., max_age: float = 2.) -> None:
        self.max_distance, self.max_age, self.next_id, self.tracks = max_distance, max_age, 1, {}
    def update(self, persons: List[Dict[str, Any]], timestamp: float) -> List[Dict[str, Any]]:
        unmatched = set(self.tracks)
        for person in persons:
            c = _center(person); best = min(unmatched, key=lambda pid: _distance(c, self.tracks[pid]["center"]), default=None)
            if best is not None and _distance(c, self.tracks[best]["center"]) <= self.max_distance:
                pid = best; unmatched.remove(pid)
            else:
                pid, self.next_id = self.next_id, self.next_id + 1
            person["person_id"], person["center"], person["timestamp"] = pid, c, timestamp
            self.tracks[pid] = {"center": c, "last_seen": timestamp, "bbox": person["bbox"]}
        self.tracks = {pid: data for pid, data in self.tracks.items() if timestamp-data["last_seen"] <= self.max_age}
        return persons


class DetectionManager:
    """Loads person, pose, and optional custom weapon models once."""
    def __init__(self) -> None:
        self.device = self._device()
        self.person_model, self.pose_model, self.weapon_model = None, None, None
        self.error, self.weapon_error = None, None
    @staticmethod
    def _device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception: return "cpu"
    def load_models(self) -> None:
        try:
            from ultralytics import YOLO
            os.makedirs("models", exist_ok=True)
            self.person_model, self.pose_model = YOLO(MODEL_PATH), YOLO(POSE_MODEL_PATH)
        except Exception as exc:
            self.error = str(exc); logger.error("YOLO model loading failed: %s", exc)
        if os.path.exists(WEAPON_MODEL_PATH):
            try:
                from ultralytics import YOLO
                self.weapon_model = YOLO(WEAPON_MODEL_PATH)
            except Exception as exc:
                self.weapon_error = str(exc); logger.error("Weapon model loading failed: %s", exc)
        else:
            self.weapon_error = f"Weapon model not found: {WEAPON_MODEL_PATH}"
    def infer(self, frame: Any, pose_enabled: bool, detection_enabled: bool) -> List[Dict[str, Any]]:
        if not detection_enabled and not pose_enabled: return []
        model = self.pose_model if pose_enabled else self.person_model
        if model is None: return []
        import cv2
        h, w = frame.shape[:2]; small = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
        result = model.predict(source=small, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD,
                               imgsz=(INFERENCE_HEIGHT, INFERENCE_WIDTH), device=self.device, verbose=False)[0]
        if result.boxes is None: return []
        xy = result.keypoints.xy.cpu().numpy() if pose_enabled and result.keypoints is not None else []
        cf = result.keypoints.conf.cpu().numpy() if pose_enabled and result.keypoints is not None and result.keypoints.conf is not None else []
        sx, sy, people = w / INFERENCE_WIDTH, h / INFERENCE_HEIGHT, []
        for i, box in enumerate(result.boxes):
            if int(box.cls[0]) != PERSON_CLASS_ID: continue
            x1, y1, x2, y2 = box.xyxy[0].tolist(); points = {}
            if i < len(xy):
                for j, name in enumerate(KEYPOINT_NAMES):
                    if j < len(xy[i]): points[name] = {"x": float(xy[i][j][0]*sx), "y": float(xy[i][j][1]*sy), "confidence": float(cf[i][j]) if i < len(cf) else 0.0}
            people.append({"person_id": None, "bbox": (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)), "confidence": float(box.conf[0]), "keypoints": points})
        return people

    def infer_weapons(self, frame: Any) -> List[Dict[str, Any]]:
        """Run the separately trained weapon detector, if its weights are installed."""
        if self.weapon_model is None: return []
        import cv2
        h, w = frame.shape[:2]; small = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
        output = self.weapon_model.predict(source=small, conf=WEAPON_CONFIDENCE_THRESHOLD,
                                           imgsz=(INFERENCE_HEIGHT, INFERENCE_WIDTH), device=self.device,
                                           verbose=False)[0]
        if output.boxes is None: return []
        names = getattr(self.weapon_model, "names", {})
        detections = []
        for box in output.boxes:
            class_id = int(box.cls[0]); x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
            detections.append({"bbox": (int(x1*w/INFERENCE_WIDTH), int(y1*h/INFERENCE_HEIGHT),
                                         int(x2*w/INFERENCE_WIDTH), int(y2*h/INFERENCE_HEIGHT)),
                               "class_id": class_id, "label": str(label), "confidence": float(box.conf[0])})
        return detections


class CameraManager:
    """M3-derived latest-frame capture, reconnect, and one AI-worker pipeline."""
    def __init__(self, detection: DetectionManager, interactions: InteractionManager) -> None:
        self.detection, self.interactions, self.tracker = detection, interactions, TrackingManager()
        self.stop_event, self.capture_stop, self.paused = threading.Event(), threading.Event(), threading.Event()
        self.camera_lock, self.frame_lock, self.result_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self.camera, self.source, self.latest_frame, self.frame_id = None, None, None, 0
        self.latest_result = {"persons": [], "weapons": [], "timestamp": 0., "inference_time_ms": 0., "error": None}
        self.capture_thread = self.worker_thread = None
        self.capture_fps, self.detection_fps, self.pose_fps = RollingFPS(), RollingFPS(), RollingFPS()
        self.detection_enabled, self.pose_enabled = True, True
    def start(self, source: Any) -> None:
        self.stop_camera(); self.stop_event.clear(); self.capture_stop.clear(); self.source = int(source) if str(source).isdigit() else source
        import cv2
        camera = cv2.VideoCapture(self.source); camera.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFER_SIZE)
        if not camera.isOpened(): camera.release(); raise RuntimeError("Camera source could not be opened")
        with self.camera_lock: self.camera = camera
        self.capture_thread = threading.Thread(target=self._capture, name="camera-capture", daemon=True)
        self.worker_thread = threading.Thread(target=self._worker, name="detection-pose-tracking", daemon=True)
        self.capture_thread.start(); self.worker_thread.start(); self.interactions.start_worker()
    def _capture(self) -> None:
        while not self.stop_event.is_set():
            with self.camera_lock: cam = self.camera
            ok, frame = cam.read() if cam is not None else (False, None)
            if ok and frame is not None:
                with self.frame_lock: self.latest_frame, self.frame_id = frame, self.frame_id + 1
                self.capture_fps.tick(); continue
            if self.stop_event.wait(RECONNECT_DELAY): break
            with self.camera_lock:
                if self.camera is not None: self.camera.release()
                import cv2; self.camera = cv2.VideoCapture(self.source)
    def _worker(self) -> None:
        last, examined = -1, 0
        while not self.stop_event.is_set():
            if self.paused.is_set(): time.sleep(.05); continue
            with self.frame_lock: frame, fid = self.latest_frame, self.frame_id
            if frame is None or fid == last: time.sleep(.004); continue
            last, examined = fid, examined + 1
            if DETECTION_INTERVAL > 1 and examined % DETECTION_INTERVAL: continue
            started, now = time.perf_counter(), time.time()
            try:
                people = self.tracker.update(self.detection.infer(frame, self.pose_enabled, self.detection_enabled), now)
                weapons = self.detection.infer_weapons(frame)
                result = {"persons": people, "weapons": weapons, "timestamp": now,
                          "inference_time_ms": (time.perf_counter()-started)*1000, "error": None}
                with self.result_lock: self.latest_result = result
                self.interactions.submit_latest(people, now); self.detection_fps.tick()
                if self.pose_enabled: self.pose_fps.tick()
            except Exception as exc:
                with self.result_lock: self.latest_result = {"persons": [], "weapons": [], "timestamp": now, "inference_time_ms": 0., "error": str(exc)}
    def snapshot(self) -> Tuple[Any, Dict[str, Any]]:
        with self.frame_lock: frame = None if self.latest_frame is None else self.latest_frame.copy()
        with self.result_lock:
            result = dict(self.latest_result)
            result["persons"] = list(self.latest_result["persons"])
            result["weapons"] = list(self.latest_result.get("weapons", []))
        return frame, result
    def stop_camera(self) -> None:
        self.stop_event.set()
        for thread in (self.capture_thread, self.worker_thread):
            if thread and thread.is_alive(): thread.join(timeout=2)
        with self.camera_lock:
            if self.camera is not None: self.camera.release(); self.camera = None


class PerformanceMonitor:
    """Collects inexpensive pipeline counters for the Performance tab."""
    def values(self, camera: CameraManager, interaction: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        perf = (interaction or {}).get("performance", {})
        return {"Camera FPS": camera.capture_fps.value(), "Detection FPS": camera.detection_fps.value(), "Pose FPS": camera.pose_fps.value(), "Tracking FPS": camera.detection_fps.value(), "Interaction FPS": perf.get("processing_fps", 0.), "Average Processing ms": perf.get("average_analysis_ms", 0.)}


class MainApplication:
    """One Tk GUI owning the single camera, YOLO, tracking, and interaction pipelines."""
    def __init__(self) -> None:
        self.root = tk.Tk(); self.root.title("AI Surveillance / Analysis System"); self.root.geometry("1300x820")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.interactions, self.detector = InteractionManager(), DetectionManager(); self.detector.load_models()
        self.camera, self.monitor = CameraManager(self.detector, self.interactions), PerformanceMonitor()
        self.status = StringVar(value="Ready" if not self.detector.error else f"Model error: {self.detector.error}")
        self.source = StringVar(value=DEFAULT_CAMERA_URL); self.pause_text = StringVar(value="Pause")
        self.overlay = {name: BooleanVar(value=True) for name in ("Boxes", "IDs", "Pose", "Weapon Boxes", "Interaction Lines", "Distance", "Proximity", "Approach", "Following", "Facing")}
        self._build(); self.root.after(PREVIEW_UPDATE_MS, self.refresh)
    def _build(self) -> None:
        ttk.Label(self.root, text="AI SURVEILLANCE / ANALYSIS SYSTEM", font=("Arial", 16, "bold")).pack(pady=7)
        ttk.Label(self.root, textvariable=self.status).pack()
        bar = ttk.Frame(self.root); bar.pack(fill="x", padx=8, pady=5)
        ttk.Label(bar, text="Camera source:").pack(side="left"); ttk.Entry(bar, textvariable=self.source, width=38).pack(side="left", padx=4)
        ttk.Button(bar, text="Start Camera", command=self.start).pack(side="left", padx=3); ttk.Button(bar, text="Stop Camera", command=self.camera.stop_camera).pack(side="left", padx=3); ttk.Button(bar, textvariable=self.pause_text, command=self.pause).pack(side="left", padx=3)
        self.book = ttk.Notebook(self.root); self.book.pack(fill="both", expand=True, padx=8, pady=5)
        self.tabs = {name: ttk.Frame(self.book) for name in ("Dashboard", "Detection", "Pose", "Tracking", "Proximity", "Distance", "Movement", "Approach", "Following", "Facing", "Hand Interaction", "Temporal", "Interaction State", "Performance", "Settings")}
        for name, tab in self.tabs.items(): self.book.add(tab, text=name)
        self.dashboard = StringVar(); ttk.Label(self.tabs["Dashboard"], textvariable=self.dashboard, justify="left", font=("Consolas", 11)).pack(anchor="nw", padx=18, pady=18)
        self.preview = ttk.Label(self.tabs["Detection"], text="Start camera to show the live preview."); self.preview.pack(fill="both", expand=True)
        ttk.Label(self.tabs["Pose"], text="Pose overlays appear in the Detection preview when enabled.").pack(padx=12, pady=12)
        self.tables = {}
        for name in ("Tracking", "Proximity", "Distance", "Movement", "Approach", "Following", "Facing", "Hand Interaction", "Temporal", "Interaction State"):
            tree = ttk.Treeview(self.tabs[name], show="headings"); tree.pack(fill="both", expand=True, padx=8, pady=8); self.tables[name] = tree
        self._columns("Tracking", ("ID", "X", "Y", "Width", "Height", "Speed")); self._columns("Movement", ("ID", "Speed", "VX", "VY", "Direction"))
        paircols = ("A", "B", "Distance", "Proximity", "Approach", "Following", "Facing", "Hand", "Duration", "Score")
        for n in ("Proximity", "Distance", "Approach", "Following", "Facing", "Hand Interaction", "Temporal", "Interaction State"): self._columns(n, paircols)
        self.performance = StringVar(); ttk.Label(self.tabs["Performance"], textvariable=self.performance, justify="left", font=("Consolas", 11)).pack(anchor="nw", padx=18, pady=18)
        settings = self.tabs["Settings"]
        ttk.Label(settings, text="Analyzer controls (disabled analyzers are not run)").pack(anchor="w", padx=10, pady=8)
        for name in self.interactions.enabled:
            var = BooleanVar(value=True); ttk.Checkbutton(settings, text=name.title(), variable=var, command=lambda n=name, v=var: self.interactions.enabled.__setitem__(n, v.get())).pack(anchor="w", padx=18)
        ttk.Separator(settings).pack(fill="x", pady=8)
        ttk.Label(settings, text="Preview overlays").pack(anchor="w", padx=10)
        for name, var in self.overlay.items(): ttk.Checkbutton(settings, text=name, variable=var).pack(anchor="w", padx=18)
    def _columns(self, name: str, cols: Tuple[str, ...]) -> None:
        tree = self.tables[name]; tree["columns"] = cols
        for col in cols: tree.heading(col, text=col); tree.column(col, width=110, anchor="center")
    def start(self) -> None:
        try: self.camera.start(self.source.get()); self.status.set("Camera connected; detection, pose, tracking, and interaction active")
        except Exception as exc: self.status.set(f"Camera error: {exc}"); messagebox.showerror("Camera", str(exc))
    def pause(self) -> None:
        if self.camera.paused.is_set(): self.camera.paused.clear(); self.pause_text.set("Pause")
        else: self.camera.paused.set(); self.pause_text.set("Resume")
    def _draw(self, frame: Any, result: Mapping[str, Any], interactions: Optional[Mapping[str, Any]]) -> Any:
        import cv2
        for p in result["persons"]:
            x1,y1,x2,y2=p["bbox"]
            if self.overlay["Boxes"].get(): cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),2)
            if self.overlay["IDs"].get(): cv2.putText(frame,f"ID {p['person_id']} {p['confidence']:.2f}",(x1,max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,0),1)
            if self.overlay["Pose"].get():
                points=[_keypoint(p,n) for n in KEYPOINT_NAMES]
                for a,b in SKELETON:
                    if points[a] and points[b]: cv2.line(frame,tuple(map(int,points[a])),tuple(map(int,points[b])),(255,180,0),1)
        if self.overlay["Weapon Boxes"].get():
            for weapon in result.get("weapons", []):
                x1, y1, x2, y2 = weapon["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"WEAPON: {weapon['label']} {weapon['confidence']:.2f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        if interactions and self.overlay["Interaction Lines"].get():
            people={p["person_id"]:p for p in result["persons"]}
            for pair in interactions["pairs"]:
                if pair["person_a"] in people and pair["person_b"] in people:
                    a,b=tuple(map(int,_center(people[pair["person_a"]]))),tuple(map(int,_center(people[pair["person_b"]])))
                    cv2.line(frame,a,b,(0,180,255),1)
                    if self.overlay["Proximity"].get(): cv2.putText(frame,pair["proximity_level"],((a[0]+b[0])//2,(a[1]+b[1])//2),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,180,255),1)
        return frame
    def _update_table(self, name: str, rows: List[Tuple[Any, ...]]) -> None:
        tree=self.tables[name]
        for iid, row in enumerate(rows):
            iid=str(iid)
            if tree.exists(iid): tree.item(iid, values=row)
            else: tree.insert("", "end", iid=iid, values=row)
        for iid in tree.get_children()[len(rows):]: tree.delete(iid)
    def refresh(self) -> None:
        frame,result=self.camera.snapshot(); inter=self.interactions.latest_result(); persons=result["persons"]
        perf=self.monitor.values(self.camera,inter); counts=inter or {"candidate_pairs":0,"nearby_pairs":0,"active_pair_states":0,"pairs":[]}
        weapon_count = len(result.get("weapons", []))
        self.dashboard.set(f"SYSTEM DASHBOARD\n\nCamera: {'CONNECTED' if self.camera.camera else 'DISCONNECTED'}\nDetection: {'ACTIVE' if self.camera.detection_enabled else 'OFF'}\nPose: {'ACTIVE' if self.camera.pose_enabled else 'OFF'}\nWeapon Detection: {'ACTIVE' if self.detector.weapon_model is not None else 'MODEL NOT INSTALLED'}\nTracking: ACTIVE\nInteraction: ACTIVE\n\nPersons: {len(persons)}\nWeapons: {weapon_count}\nCandidate Pairs: {counts['candidate_pairs']}\nNearby Pairs: {counts['nearby_pairs']}\nActive Pair States: {counts['active_pair_states']}\n\nDetection FPS: {perf['Detection FPS']:.1f}\nPose FPS: {perf['Pose FPS']:.1f}\nInteraction FPS: {perf['Interaction FPS']:.1f}\nProcessing Time: {result['inference_time_ms']:.1f} ms")
        self.performance.set("\n".join(f"{k}: {v:.2f}" for k,v in perf.items()))
        motions=self.interactions.movement.update([], result["timestamp"]) if False else {}
        self._update_table("Tracking", [(p["person_id"],int(_center(p)[0]),int(_center(p)[1]),p["bbox"][2]-p["bbox"][0],p["bbox"][3]-p["bbox"][1],"-") for p in persons])
        pairs=counts["pairs"]; pairrows=[(p["person_a"],p["person_b"],f"{p['normalized_distance']:.2f}",p["proximity_level"],"YES" if p.get("approaching") else "NO",f"{p.get('following_score',0):.2f}",p.get("facing_status","UNKNOWN"),"YES" if p.get("hand_near_body") else "NO",f"{p.get('close_duration',0):.1f}s",f"{p['interaction_score']:.2f}") for p in pairs]
        for name in ("Proximity","Distance","Approach","Following","Facing","Hand Interaction","Temporal","Interaction State"): self._update_table(name,pairrows)
        self._update_table("Movement", [(p["person_id"],"-","-","-","-") for p in persons])
        if frame is not None:
            try:
                from PIL import Image, ImageTk
                import cv2
                drawn=self._draw(frame,result,inter); drawn=cv2.cvtColor(drawn,cv2.COLOR_BGR2RGB); drawn=cv2.resize(drawn,(850,480))
                photo=ImageTk.PhotoImage(Image.fromarray(drawn)); self.preview.configure(image=photo,text=""); self.preview.image=photo
            except Exception as exc: self.preview.configure(text=f"Preview unavailable: {exc}")
        if not self.camera.stop_event.is_set() or self.root.winfo_exists(): self.root.after(PREVIEW_UPDATE_MS,self.refresh)
    def run(self) -> None: self.root.mainloop()
    def close(self) -> None:
        self.camera.stop_camera(); self.interactions.stop_worker(); self.root.destroy()


def main() -> None:
    """Start the complete one-file application."""
    MainApplication().run()


if __name__ == "__main__":
    main()
