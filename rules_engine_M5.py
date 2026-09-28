"""Module 5: explainable, temporal safety-risk assessment for M4 interactions.

This standalone prototype consumes the *feature dictionaries* emitted by
``interactive_analyser_M4.py``; it never imports or modifies that module.  It
does not infer guilt, intent, harassment, or a confirmed crime.  Scores flag
potentially unsafe interaction patterns for human review.

M4 adapter map: person_a/person_b, normalized_distance, proximity_level,
close_duration, approach_score/approaching, following_score/following,
a_speed/b_speed, facing_score, hand_near_body_score/hand_near_body,
interaction_score, target_moving_away and detection confidence (when present).
M4 does not expose overlap, acceleration, direction changes, or group geometry;
those rules are marked unavailable unless a future producer supplies them.
"""
from __future__ import annotations

import argparse
import os
import copy
import logging
import math
import queue
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from itertools import combinations
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger("rules_engine_M5")
PairKey = Tuple[int, int]


# Starting parameters only: calibrate against representative, reviewed video.
@dataclass
class M5Config:
    # Evaluation and temporal memory
    history_size: int = 90
    evaluation_interval_seconds: float = 0.0
    default_fps: float = 0.0                 # never silently assume 30 FPS
    max_frame_delta_seconds: float = 1.5
    short_tracking_gap_seconds: float = 2.0
    stale_pair_timeout_seconds: float = 45.0
    episode_gap_seconds: float = 4.0
    event_history_size: int = 100
    # Smoothness / false-positive controls
    ema_alpha: float = 0.28
    median_window: int = 5
    close_persistence_seconds: float = 2.5
    duration_full_seconds: float = 15.0
    following_persistence_seconds: float = 2.5
    movement_persistence_seconds: float = 1.0
    hand_persistence_seconds: float = 1.0
    overlap_persistence_seconds: float = 1.0
    escalation_window_seconds: float = 3.0
    escalation_distance_delta: float = 0.55
    rapid_approach_score: float = 0.65
    high_speed: float = 70.0
    high_speed_full: float = 180.0
    overlap_meaningful: float = 0.18
    # Repeated approach/retreat
    cycle_window_seconds: float = 18.0
    cycles_for_full_signal: int = 3
    trend_deadband: float = 0.08
    # Group adapter: only activated by supplied nearby_person_count/group_* fields
    group_min_nearby_people: int = 2
    # Central weights. Contributions are weight * normalized signal.
    weights: Dict[str, float] = field(default_factory=lambda: {
        "persistent_proximity": 18.0, "interaction_duration": 9.0,
        "following": 17.0, "rapid_approach": 10.0, "proximity_escalation": 9.0,
        "movement_context": 8.0, "body_overlap": 10.0, "hand_proximity": 7.0,
        "repeated_approach": 8.0, "group_surrounding": 8.0,
    })
    combination_bonuses: Dict[str, float] = field(default_factory=lambda: {
        "FOLLOWING_PLUS_DURATION": 8.0, "APPROACH_CLOSE_MOVEMENT": 7.0,
        "OVERLAP_HAND_MOVEMENT": 7.0, "GROUP_PROXIMITY_PERSISTENCE": 6.0,
        "FACING_CONTEXT": 2.0,
    })
    max_combination_bonus: float = 18.0
    raw_score_cap: float = 100.0
    # State machine: enter threshold is deliberately above exit threshold.
    enter_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 12.0, "MONITORING": 26.0, "SUSPICIOUS": 50.0, "HIGH_RISK": 75.0,
    })
    exit_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "LOW": 6.0, "MONITORING": 18.0, "SUSPICIOUS": 39.0, "HIGH_RISK": 63.0,
    })
    high_risk_min_persistence_seconds: float = 1.5
    strong_combination_minimum: int = 2
    alert_cooldown_seconds: float = 20.0
    gui_update_interval_ms: int = 100
    preview_update_interval_ms: int = 50
    reconnect_delay_seconds: float = 2.0
    capture_buffer_size: int = 1
    debug: bool = False
    debug_interval_seconds: float = 2.0


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    LOW = "LOW"
    MONITORING = "MONITORING"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"
    RESOLVED = "RESOLVED"


@dataclass
class Episode:
    episode_id: str
    pair: PairKey
    started_at: float
    latest_at: float
    peak_raw_risk: float = 0.0
    peak_smoothed_risk: float = 0.0
    risk_sum: float = 0.0
    observations: int = 0
    state: RiskState = RiskState.NORMAL
    triggered_rules: set[str] = field(default_factory=set)
    reason_codes: set[str] = field(default_factory=set)
    alerted_states: set[RiskState] = field(default_factory=set)
    ended_at: Optional[float] = None

    def update(self, timestamp: float, raw: float, smooth: float, state: RiskState,
               triggered: Iterable[str]) -> None:
        self.latest_at, self.peak_raw_risk = timestamp, max(self.peak_raw_risk, raw)
        self.peak_smoothed_risk = max(self.peak_smoothed_risk, smooth)
        self.risk_sum, self.observations, self.state = self.risk_sum + smooth, self.observations + 1, state
        self.triggered_rules.update(triggered); self.reason_codes.update(triggered)

    def snapshot(self) -> Dict[str, Any]:
        return {"episode_id": self.episode_id, "pair": self.pair, "start_timestamp": self.started_at,
                "latest_timestamp": self.latest_at, "end_timestamp": self.ended_at,
                "duration_seconds": max(0.0, (self.ended_at or self.latest_at) - self.started_at),
                "peak_raw_risk": self.peak_raw_risk, "peak_smoothed_risk": self.peak_smoothed_risk,
                "average_risk": self.risk_sum / self.observations if self.observations else 0.0,
                "state": self.state.value, "triggered_rules": sorted(self.triggered_rules),
                "reason_codes": sorted(self.reason_codes)}


@dataclass
class PairMemory:
    pair: PairKey
    first_seen: float
    last_seen: float
    last_evaluated: float = 0.0
    active_duration: float = 0.0
    state: RiskState = RiskState.NORMAL
    risk_ema: float = 0.0
    last_alert_at: float = float("-inf")
    episode: Optional[Episode] = None
    distance: Deque[Tuple[float, float]] = field(default_factory=deque)
    trends: Deque[Tuple[float, str]] = field(default_factory=deque)
    signals: Dict[str, Deque[Tuple[float, float]]] = field(default_factory=dict)
    risks: Deque[float] = field(default_factory=deque)


def _finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ramp(value: Optional[float], start: float, full: float) -> float:
    if value is None or full <= start:
        return 0.0
    return _clamp((value - start) / (full - start))


def canonical_pair(a: Any, b: Any) -> Optional[PairKey]:
    try:
        aa, bb = int(a), int(b)
        return (aa, bb) if aa != bb and aa < bb else (bb, aa) if aa != bb else None
    except (TypeError, ValueError):
        return None


class FeatureAdapter:
    """Read M4-compatible mappings without fabricating fields or semantics."""
    AVAILABLE = frozenset(("persistent_proximity", "interaction_duration", "following",
                           "rapid_approach", "proximity_escalation", "movement_context",
                           "hand_proximity", "repeated_approach"))

    @staticmethod
    def pair(feature: Mapping[str, Any]) -> Optional[PairKey]:
        return canonical_pair(feature.get("person_a"), feature.get("person_b"))

    @staticmethod
    def confidence(feature: Mapping[str, Any]) -> Optional[float]:
        values = [_finite(feature.get(k)) for k in ("detection_confidence", "confidence", "a_confidence", "b_confidence")]
        values = [v for v in values if v is not None]
        return _clamp(sum(values) / len(values)) if values else None

    @staticmethod
    def overlap(feature: Mapping[str, Any]) -> Optional[float]:
        # Future producers may supply normalized IoU/overlap; M4 currently does not.
        return _finite(feature.get("body_overlap", feature.get("bbox_overlap")))


class RulesEngineM5:
    """CPU-light per-pair rule engine. Call ``evaluate`` once per M4 pair output."""
    def __init__(self, config: Optional[M5Config] = None) -> None:
        self.config = config or M5Config()
        self._states: Dict[PairKey, PairMemory] = {}
        self._events: Deque[Dict[str, Any]] = deque(maxlen=self.config.event_history_size)
        self._lock = threading.RLock()
        self._last_debug = 0.0

    @property
    def recent_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def reset(self) -> None:
        with self._lock:
            self._states.clear(); self._events.clear()

    def _state_for(self, pair: PairKey, ts: float) -> PairMemory:
        state = self._states.get(pair)
        if state is None or ts < state.last_seen:  # a source restart/rewind creates clean timing.
            state = PairMemory(pair, ts, ts)
            self._states[pair] = state
        return state

    def _history(self, state: PairMemory, name: str) -> Deque[Tuple[float, float]]:
        if name not in state.signals:
            state.signals[name] = deque(maxlen=self.config.history_size)
        return state.signals[name]

    def _persistence(self, state: PairMemory, name: str, ts: float, seconds: float) -> float:
        history = self._history(state, name)
        relevant = [v for t, v in history if t >= ts - seconds]
        return sum(relevant) / len(relevant) if relevant else 0.0

    def _trend_cycles(self, state: PairMemory, ts: float) -> int:
        recent = [(t, v) for t, v in state.trends if t >= ts - self.config.cycle_window_seconds]
        # count completed APPROACH -> RETREAT -> APPROACH triplets, no repeated frame counts.
        compact: List[str] = []
        for _, trend in recent:
            if not compact or compact[-1] != trend:
                compact.append(trend)
        return sum(1 for i in range(len(compact) - 2) if compact[i:i + 3] == ["APPROACH", "RETREAT", "APPROACH"])

    def _signals(self, feature: Mapping[str, Any], state: PairMemory, ts: float) -> Tuple[Dict[str, float], Dict[str, bool]]:
        cfg = self.config
        dist = _finite(feature.get("normalized_distance"))
        level = str(feature.get("proximity_level", "")).upper()
        if dist is not None:
            state.distance.append((ts, max(0.0, dist)))
        close_now = 1.0 if level in ("CLOSE", "VERY_CLOSE") else _ramp(3.0 - dist if dist is not None else None, 0.6, 2.15)
        self._history(state, "close").append((ts, close_now))
        supplied_duration = _finite(feature.get("close_duration"))
        close_duration = supplied_duration if supplied_duration is not None else state.active_duration * self._persistence(state, "close", ts, cfg.duration_full_seconds)
        proximity = close_now * _ramp(close_duration, cfg.close_persistence_seconds, cfg.duration_full_seconds)
        duration = _ramp(state.active_duration, cfg.close_persistence_seconds, cfg.duration_full_seconds)

        following_raw = _clamp(_finite(feature.get("following_score")) or 0.0)
        self._history(state, "following").append((ts, following_raw))
        following = following_raw * _ramp(state.active_duration, cfg.following_persistence_seconds, cfg.duration_full_seconds)

        approach_raw = _clamp(_finite(feature.get("approach_score")) or 0.0)
        self._history(state, "approach").append((ts, approach_raw))
        rapid = _ramp(approach_raw, cfg.rapid_approach_score, 1.0) * _ramp(state.active_duration, .5, 3.0)

        escalation = 0.0
        old = [d for t, d in state.distance if t <= ts - .2 and t >= ts - cfg.escalation_window_seconds]
        if dist is not None and old:
            escalation = _ramp(statistics.median(old) - dist, cfg.escalation_distance_delta, cfg.escalation_distance_delta * 2.5)

        speeds = [_finite(feature.get("a_speed")), _finite(feature.get("b_speed")), _finite(feature.get("speed"))]
        speed = max((v for v in speeds if v is not None), default=None)
        movement_raw = _ramp(speed, cfg.high_speed, cfg.high_speed_full)
        self._history(state, "movement").append((ts, movement_raw))
        movement = movement_raw * close_now * self._persistence(state, "movement", ts, cfg.movement_persistence_seconds)

        overlap_value = FeatureAdapter.overlap(feature)
        overlap_available = overlap_value is not None
        overlap = _ramp(overlap_value, cfg.overlap_meaningful, 0.65) if overlap_available else 0.0
        self._history(state, "overlap").append((ts, overlap))
        overlap *= self._persistence(state, "overlap", ts, cfg.overlap_persistence_seconds) * close_now

        hand_available = "hand_near_body_score" in feature or "hand_near_body" in feature
        hand_raw = _clamp(_finite(feature.get("hand_near_body_score")) or (1.0 if feature.get("hand_near_body") else 0.0))
        self._history(state, "hand").append((ts, hand_raw))
        hand = hand_raw * close_now * self._persistence(state, "hand", ts, cfg.hand_persistence_seconds)
        facing_value = _finite(feature.get("facing_score"))
        facing_available = facing_value is not None
        # M4's shoulder alignment is contextual only: it receives no direct weight.
        facing = _clamp(facing_value) if facing_available else 0.0

        trend = "APPROACH" if approach_raw >= cfg.trend_deadband else "RETREAT" if str(feature.get("distance_trend", "")).upper() == "INCREASING" else "STABLE"
        if not state.trends or state.trends[-1][1] != trend:
            state.trends.append((ts, trend))
        repeated = _ramp(float(self._trend_cycles(state, ts)), 0.0, float(cfg.cycles_for_full_signal))

        count = _finite(feature.get("nearby_person_count", feature.get("group_nearby_count")))
        group_available = count is not None or "group_surrounding_score" in feature
        group = _clamp(_finite(feature.get("group_surrounding_score")) or _ramp(count, cfg.group_min_nearby_people, cfg.group_min_nearby_people + 2))
        group *= close_now * _ramp(state.active_duration, cfg.close_persistence_seconds, cfg.duration_full_seconds)

        return ({"persistent_proximity": proximity, "interaction_duration": duration,
                 "following": following, "rapid_approach": rapid, "proximity_escalation": escalation,
                 "movement_context": movement, "body_overlap": overlap, "hand_proximity": hand,
                 "repeated_approach": repeated, "group_surrounding": group, "facing_context": facing},
                {"body_overlap": overlap_available, "hand_proximity": hand_available,
                 "group_surrounding": group_available, "facing_context": facing_available})

    def _state_transition(self, old: RiskState, score: float, duration: float, combinations: int) -> RiskState:
        enter, exit = self.config.enter_thresholds, self.config.exit_thresholds
        # Escalation requires corroboration and persistence. De-escalation uses hysteresis.
        if old == RiskState.HIGH_RISK:
            return RiskState.HIGH_RISK if score >= exit["HIGH_RISK"] else RiskState.SUSPICIOUS
        high_allowed = duration >= self.config.high_risk_min_persistence_seconds and combinations >= self.config.strong_combination_minimum
        if score >= enter["HIGH_RISK"] and high_allowed: return RiskState.HIGH_RISK
        if old == RiskState.SUSPICIOUS:
            return RiskState.SUSPICIOUS if score >= exit["SUSPICIOUS"] else RiskState.MONITORING
        if score >= enter["SUSPICIOUS"]: return RiskState.SUSPICIOUS
        if old == RiskState.MONITORING:
            return RiskState.MONITORING if score >= exit["MONITORING"] else RiskState.LOW
        if score >= enter["MONITORING"]: return RiskState.MONITORING
        if old == RiskState.LOW:
            return RiskState.LOW if score >= exit["LOW"] else RiskState.NORMAL
        return RiskState.LOW if score >= enter["LOW"] else RiskState.NORMAL

    @staticmethod
    def _reasons(signals: Mapping[str, float], available: Mapping[str, bool]) -> Tuple[List[str], List[str]]:
        items = (("persistent_proximity", "PERSISTENT_PROXIMITY", "Persistent close interaction"),
                 ("interaction_duration", "LONG_INTERACTION", "Prolonged interaction"),
                 ("following", "HIGH_FOLLOWING", "Sustained following pattern"),
                 ("rapid_approach", "RAPID_APPROACH", "Rapid approach trend"),
                 ("proximity_escalation", "PROXIMITY_ESCALATION", "Proximity is escalating"),
                 ("movement_context", "MOVEMENT_ESCALATION", "Elevated movement while close"),
                 ("body_overlap", "BODY_OVERLAP", "Persistent body-box overlap"),
                 ("hand_proximity", "HAND_PROXIMITY", "Persistent hand-to-body proximity"),
                 ("repeated_approach", "REPEATED_APPROACH", "Repeated approach-retreat cycles"),
                 ("group_surrounding", "GROUP_SURROUNDING", "Persistent nearby group context"))
        codes, words = [], []
        for signal, code, reason in items:
            if signals.get(signal, 0.0) >= .30 and (signal not in available or available[signal]):
                codes.append(code); words.append(reason)
        return codes, words

    def evaluate(self, feature: Mapping[str, Any], timestamp: Optional[float] = None,
                 frame_index: Optional[int] = None, fps: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Evaluate one M4 pair. Invalid/missing IDs return ``None`` safely."""
        pair = FeatureAdapter.pair(feature)
        if pair is None:
            return None
        ts = _finite(timestamp if timestamp is not None else feature.get("timestamp"))
        if ts is None:
            valid_fps = _finite(fps if fps is not None else feature.get("fps"))
            ts = frame_index / valid_fps if frame_index is not None and valid_fps and valid_fps > 0 else time.monotonic()
        with self._lock:
            state = self._state_for(pair, ts)
            gap = max(0.0, ts - state.last_seen)
            if state.last_evaluated and ts - state.last_evaluated < self.config.evaluation_interval_seconds:
                return None
            if gap > self.config.short_tracking_gap_seconds and state.episode:
                state.episode.ended_at = state.last_seen; state.episode = None; state.active_duration = 0.0
            else:
                state.active_duration += min(gap, self.config.max_frame_delta_seconds)
            state.last_seen, state.last_evaluated = ts, ts
            if state.episode is None:
                state.episode = Episode(f"M5-{pair[0]}-{pair[1]}-{uuid.uuid4().hex[:8]}", pair, ts, ts)
            signals, available = self._signals(feature, state, ts)
            contributions = {name: signals[name] * self.config.weights[name] for name in self.config.weights}
            combos: Dict[str, float] = {}
            if signals["following"] >= .55 and signals["interaction_duration"] >= .45:
                combos["FOLLOWING_PLUS_DURATION"] = self.config.combination_bonuses["FOLLOWING_PLUS_DURATION"]
            if signals["rapid_approach"] >= .45 and signals["persistent_proximity"] >= .45 and signals["movement_context"] >= .30:
                combos["APPROACH_CLOSE_MOVEMENT"] = self.config.combination_bonuses["APPROACH_CLOSE_MOVEMENT"]
            if available["body_overlap"] and signals["body_overlap"] >= .40 and signals["hand_proximity"] >= .30 and signals["movement_context"] >= .25:
                combos["OVERLAP_HAND_MOVEMENT"] = self.config.combination_bonuses["OVERLAP_HAND_MOVEMENT"]
            if available["group_surrounding"] and signals["group_surrounding"] >= .40 and signals["persistent_proximity"] >= .35:
                combos["GROUP_PROXIMITY_PERSISTENCE"] = self.config.combination_bonuses["GROUP_PROXIMITY_PERSISTENCE"]
            if available["facing_context"] and signals["facing_context"] >= .60 and (signals["hand_proximity"] >= .35 or signals["persistent_proximity"] >= .55):
                combos["FACING_CONTEXT"] = self.config.combination_bonuses["FACING_CONTEXT"]
            bonus = min(self.config.max_combination_bonus, sum(combos.values()))
            raw = _clamp(sum(contributions.values()) + bonus, 0.0, self.config.raw_score_cap)
            state.risk_ema = raw if not state.risks else self.config.ema_alpha * raw + (1 - self.config.ema_alpha) * state.risk_ema
            state.risks.append(state.risk_ema)
            median = statistics.median(list(state.risks)[-self.config.median_window:])
            smooth = _clamp((state.risk_ema + median) / 2, 0.0, 100.0)
            previous_state = state.state
            state.state = self._state_transition(previous_state, smooth, state.active_duration, len(combos))
            codes, reasons = self._reasons(signals, available)
            state.episode.update(ts, raw, smooth, state.state, codes)
            elevated = state.state in (RiskState.SUSPICIOUS, RiskState.HIGH_RISK)
            cooldown_remaining = max(0.0, self.config.alert_cooldown_seconds - (ts - state.last_alert_at))
            escalation = state.state == RiskState.HIGH_RISK and RiskState.HIGH_RISK not in state.episode.alerted_states
            alert_allowed = elevated and (cooldown_remaining <= 0 or escalation) and state.state not in state.episode.alerted_states
            if alert_allowed:
                state.last_alert_at = ts; state.episode.alerted_states.add(state.state)
            result = {"pair_ids": pair, "person_a": pair[0], "person_b": pair[1], "timestamp": ts,
                      "episode_id": state.episode.episode_id, "interaction_duration_seconds": state.active_duration,
                      "raw_features": dict(feature), "normalized_signals": signals, "signal_available": available,
                      "rule_weights": dict(self.config.weights), "rule_contributions": contributions,
                      "combination_bonuses": combos, "combination_bonus_total": bonus,
                      "raw_risk_score": raw, "smoothed_risk_score": smooth, "suspiciousness_score": smooth,
                      "risk_state": state.state.value, "previous_risk_state": previous_state.value,
                      "triggered_rule_ids": codes, "reason_codes": codes, "reasons": reasons,
                      "detection_confidence": FeatureAdapter.confidence(feature),
                      "interaction_confidence": _finite(feature.get("interaction_score")),
                      "alert_generated": alert_allowed, "alert_suppressed": elevated and not alert_allowed,
                      "cooldown_remaining_seconds": cooldown_remaining, "next_alert_allowed_at": max(ts, state.last_alert_at + self.config.alert_cooldown_seconds),
                      "episode": state.episode.snapshot()}
            if alert_allowed:
                self._events.appendleft({"timestamp": ts, "pair_ids": pair, "episode_id": state.episode.episode_id,
                                          "score": smooth, "state": state.state.value, "reasons": reasons})
            if self.config.debug and ts - self._last_debug >= self.config.debug_interval_seconds:
                LOGGER.info("M5 %s score %.1f %s rules=%s", pair, smooth, state.state.value, codes); self._last_debug = ts
            return result

    def evaluate_batch(self, m4_result: Mapping[str, Any], timestamp: Optional[float] = None,
                       fps: Optional[float] = None) -> Dict[str, Any]:
        ts = timestamp if timestamp is not None else _finite(m4_result.get("timestamp"))
        results = [r for item in m4_result.get("pairs", ()) if (r := self.evaluate(item, ts, fps=fps))]
        self.cleanup(ts if ts is not None else time.monotonic())
        return {"timestamp": ts, "assessments": results, "highest_risk": max(results, key=lambda r: r["suspiciousness_score"], default=None), "recent_events": self.recent_events}

    def cleanup(self, timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        ts = time.monotonic() if timestamp is None else timestamp; ended = []
        with self._lock:
            for key, state in list(self._states.items()):
                if state.episode and ts - state.last_seen > self.config.episode_gap_seconds:
                    state.episode.ended_at = state.last_seen; ended.append(state.episode.snapshot()); state.episode = None
                if ts - state.last_seen > self.config.stale_pair_timeout_seconds:
                    del self._states[key]
        return ended


# GUI / pipeline: exactly one Tk root; all widgets below update only from root.after.
class LatestSlot:
    def __init__(self) -> None: self._value: Any = None; self._lock = threading.Lock()
    def put(self, value: Any) -> None:
        with self._lock: self._value = value
    def get(self) -> Any:
        with self._lock: return self._value


def normalize_source(source: str, source_kind: str) -> str:
    """Keep explicit stream endpoints intact; turn an IP Webcam base URL into /video."""
    source = source.strip()
    if source_kind == "IP Webcam" and source.startswith(("http://", "https://")):
        tail = source.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
        if tail.count(":") or tail.replace(".", "").isdigit():
            return source.rstrip("/") + "/video"
    return source


# M4-compatible standalone interaction layer. These preserve M4's image-space
# geometry, thresholds, and formulas; unlike M4, this file is self-contained.
M4_VERY_CLOSE, M4_CLOSE, M4_MODERATE = .85, 1.65, 3.0
M4_SPEED, M4_FAST_SPEED, M4_HAND = 8., 45., .80
M4_KEYPOINT_CONF = .40
def _m4_center(p: Mapping[str, Any]) -> Tuple[float, float]:
    if p.get("center") is not None: return tuple(map(float, p["center"]))
    x1,y1,x2,y2=p["bbox"]; return ((float(x1)+float(x2))/2,(float(y1)+float(y2))/2)
def _m4_height(p: Mapping[str, Any]) -> float: return max(abs(float(p["bbox"][3])-float(p["bbox"][1])),20.)
def _m4_unit(v: Tuple[float,float]) -> Optional[Tuple[float,float]]:
    n=math.hypot(*v); return (v[0]/n,v[1]/n) if n else None
def _m4_dot(a: Tuple[float,float],b: Tuple[float,float]) -> float: return a[0]*b[0]+a[1]*b[1]
def _m4_kp(p: Mapping[str,Any], name: str) -> Optional[Tuple[float,float]]:
    q=p.get("keypoints",{}).get(name)
    return (float(q["x"]),float(q["y"])) if q and _finite(q.get("confidence")) is not None and float(q["confidence"])>=M4_KEYPOINT_CONF and q.get("x") is not None and q.get("y") is not None else None

class M4CompatibleTracker:
    """The same centroid association behavior M4 implements (not ByteTrack)."""
    def __init__(self,max_distance:float=110.,max_age:float=2.): self.max_distance,self.max_age,self.next_id,self.tracks=max_distance,max_age,1,{}
    def update(self, people: List[Dict[str,Any]], ts:float) -> List[Dict[str,Any]]:
        unmatched=set(self.tracks)
        for p in people:
            c=_m4_center(p); best=min(unmatched,key=lambda i:math.dist(c,self.tracks[i]["center"]),default=None)
            if best is not None and math.dist(c,self.tracks[best]["center"])<=self.max_distance: pid=best; unmatched.remove(best)
            else: pid=self.next_id; self.next_id+=1
            p.update(person_id=pid,center=c,timestamp=ts); self.tracks[pid]={"center":c,"last_seen":ts,"bbox":p["bbox"]}
        self.tracks={i:x for i,x in self.tracks.items() if ts-x["last_seen"]<=self.max_age}; return people

class M4CompatibleDetector:
    """M4's one-time YOLO person/pose model loader and output normalization."""
    names=("nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle")
    def __init__(self): self.device,self.person_model,self.pose_model,self.error=self._device(),None,None,None
    def _device(self) -> str:
        try:
            import torch; return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception: return "cpu"
    def load_models(self) -> None:
        try:
            from ultralytics import YOLO
            self.person_model,self.pose_model=YOLO(os.path.join("models","yolo11n.pt")),YOLO(os.path.join("models","yolo11n-pose.pt"))
        except Exception as exc: self.error=str(exc); LOGGER.error("M5 model loading failed: %s",exc)
    def infer(self, frame: Any, pose: bool, enabled: bool) -> List[Dict[str,Any]]:
        if not enabled and not pose: return []
        model=self.pose_model if pose else self.person_model
        if model is None: return []
        import cv2
        h,w=frame.shape[:2]; small=cv2.resize(frame,(640,640)); out=model.predict(source=small,classes=[0],conf=.50,imgsz=(640,640),device=self.device,verbose=False)[0]
        xy=out.keypoints.xy.cpu().numpy() if pose and out.keypoints is not None else []; cf=out.keypoints.conf.cpu().numpy() if pose and out.keypoints is not None and out.keypoints.conf is not None else []
        people=[]
        if out.boxes is None: return people
        for i,b in enumerate(out.boxes):
            if int(b.cls[0])!=0: continue
            x1,y1,x2,y2=b.xyxy[0].tolist(); pts={}
            if i<len(xy):
                for j,n in enumerate(self.names):
                    if j<len(xy[i]): pts[n]={"x":float(xy[i][j][0]*w/640),"y":float(xy[i][j][1]*h/640),"confidence":float(cf[i][j]) if i<len(cf) else 0.}
            people.append({"person_id":None,"bbox":(int(x1*w/640),int(y1*h/640),int(x2*w/640),int(y2*h/640)),"confidence":float(b.conf[0]),"keypoints":pts})
        return people

class M4CompatibleInteractions:
    """Faithful M4 feature meanings with bounded per-person/pair histories."""
    def __init__(self): self.positions=defaultdict(lambda:deque(maxlen=12)); self.states={}
    def process(self, people: Sequence[Mapping[str,Any]], ts:float) -> Dict[str,Any]:
        motions={}
        for p in people:
            i=int(p["person_id"]); c=_m4_center(p); hist=self.positions[i]; vx=vy=spd=0.
            if hist and ts>hist[-1][0]: vx,vy=(c[0]-hist[-1][1][0])/(ts-hist[-1][0]),(c[1]-hist[-1][1][1])/(ts-hist[-1][0]); spd=math.hypot(vx,vy)
            hist.append((ts,c)); motions[i]={"velocity_x":vx,"velocity_y":vy,"speed":spd,"movement_direction":_m4_unit((vx,vy)),"movement_state":"FAST_MOVING" if spd>=M4_FAST_SPEED else "MOVING" if spd>=M4_SPEED else "STATIONARY"}
        output=[]
        for a,b in combinations(sorted(people,key=lambda x:int(x["person_id"])),2):
            ia,ib=int(a["person_id"]),int(b["person_id"]); ca,cb=_m4_center(a),_m4_center(b); dist=math.dist(ca,cb)/((_m4_height(a)+_m4_height(b))/2)
            if dist>M4_MODERATE: continue
            key=(ia,ib); s=self.states.setdefault(key,{"first":ts,"last":ts,"dist":deque(maxlen=10),"close":0.,"follow":0.,"approach":0.,"hand":0.}); dt=max(0.,ts-s["last"]); prev=s["dist"][-1][1] if s["dist"] else None; s["dist"].append((ts,dist)); s["last"]=ts
            level="VERY_CLOSE" if dist<=M4_VERY_CLOSE else "CLOSE" if dist<=M4_CLOSE else "MODERATE"; s["close"]+=dt if level in ("VERY_CLOSE","CLOSE") else 0.
            ds=[x[1] for x in s["dist"]]; approach=0.; approaching=False; separating=False
            if len(ds)>=3 and s["dist"][-1][0]-s["dist"][0][0]>=.6:
                tt=[x[0] for x in s["dist"]]; mt,md=statistics.mean(tt),statistics.mean(ds); den=sum((x-mt)**2 for x in tt); slope=sum((x-mt)*(y-md) for x,y in zip(tt,ds))/den if den else 0.; consistency=sum(y<x for x,y in zip(ds,ds[1:]))/(len(ds)-1); approaching=slope<-.08 and consistency>=.65; separating=slope>.08 and consistency<=.35; approach=min(1.,abs(slope)/.32)*(consistency if slope<0 else 0.)
            va,vb=motions[ia],motions[ib]; ua,ub=_m4_unit((va["velocity_x"],va["velocity_y"])),_m4_unit((vb["velocity_x"],vb["velocity_y"])); direction=max(0.,_m4_dot(ua,ub)) if ua and ub else 0.; velocity=1-min(1.,abs(va["speed"]-vb["speed"])/max(va["speed"],vb["speed"],1.)); count=min(len(self.positions[ia]),len(self.positions[ib]),12); traj=0.
            if count>=3:
                ta,tb=self.positions[ia],self.positions[ib]; xa=_m4_unit((ta[-1][1][0]-ta[-count][1][0],ta[-1][1][1]-ta[-count][1][1])); xb=_m4_unit((tb[-1][1][0]-tb[-count][1][0],tb[-1][1][1]-tb[-count][1][1])); traj=max(0.,_m4_dot(xa,xb)) if xa and xb else 0.
            stability=1-min(1.,(statistics.pstdev(ds) if len(ds)>1 else 1.)/.75); following=.28*direction+.20*velocity+.25*traj+.17*stability+.10*min(1.,(ts-s["first"])/1.); following_on=following>=.65 and count>=3; s["follow"]+=dt if following_on else 0.; s["approach"]+=dt if approaching else 0.
            shoulders=[_m4_kp(a,"left_shoulder"),_m4_kp(a,"right_shoulder"),_m4_kp(b,"left_shoulder"),_m4_kp(b,"right_shoulder")]; face=None
            if all(shoulders):
                def align(p,o):
                    l,r=_m4_kp(p,"left_shoulder"),_m4_kp(p,"right_shoulder"); axis=_m4_unit((r[0]-l[0],r[1]-l[1])); mid=((l[0]+r[0])/2,(l[1]+r[1])/2); toward=_m4_unit((o[0]-mid[0],o[1]-mid[1])); return abs(_m4_dot((-axis[1],axis[0]),toward))
                face=(align(a,cb)+align(b,ca))/2
            def hand(actor,target):
                body=[x for n in ("left_shoulder","right_shoulder","left_hip","right_hip") if (x:=_m4_kp(target,n))]; wrists=[x for n in ("left_wrist","right_wrist") if (x:=_m4_kp(actor,n))]; vals=[math.dist(w,q)/_m4_height(target) for w in wrists for q in body]; return max(0.,1-min(vals)/M4_HAND) if vals else 0.
            hs=max(hand(a,b),hand(b,a)); s["hand"]+=dt if hs>0 else 0.; persistence=min(1.,(ts-s["first"])/3); prox=max(0.,1-dist/M4_MODERATE)
            output.append({"person_a":ia,"person_b":ib,"pixel_distance":math.dist(ca,cb),"normalized_distance":dist,"previous_distance":prev,"distance_delta":None if prev is None else dist-prev,"proximity_level":level,"approaching":approaching,"separating":separating,"distance_trend":"DECREASING" if approaching else "INCREASING" if separating else "STABLE","approach_score":approach,"following_score":following,"following":following_on,"a_speed":va["speed"],"b_speed":vb["speed"],"facing_score":face,"facing_each_other":None if face is None else face>=.60,"facing_status":"UNKNOWN" if face is None else "FACING" if face>=.60 else "NOT_FACING","hand_near_body_score":hs,"hand_near_body":hs>0,"close_duration":s["close"],"approach_duration":s["approach"],"following_duration":s["follow"],"hand_proximity_duration":s["hand"],"interaction_score":min(1.,.24*prox+.18*approach+.18*following+.14*(face or 0)+.16*hs+.10*persistence),"detection_confidence":(float(a.get("confidence",0))+float(b.get("confidence",0)))/2})
        self.states={k:v for k,v in self.states.items() if ts-v["last"]<=3.}; return {"timestamp":ts,"tracked_persons":len(people),"pairs":output}


class M5Application:
    """Three independently movable Toplevel windows backed by a latest-result pipeline."""
    def __init__(self, config: Optional[M5Config] = None, feature_source: Optional[Callable[[], Mapping[str, Any]]] = None) -> None:
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.config = tk, ttk, config or M5Config()
        self.root = tk.Tk(); self.root.withdraw(); self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.engine, self.feature_source = RulesEngineM5(self.config), feature_source
        self.detector, self.tracker, self.interactions = M4CompatibleDetector(), M4CompatibleTracker(), M4CompatibleInteractions()
        self.stop_event, self.paused = threading.Event(), threading.Event()
        self.results, self.frames, self.processed_frames, self.diagnostics = LatestSlot(), LatestSlot(), LatestSlot(), LatestSlot()
        self.worker: Optional[threading.Thread] = None; self.capture_worker: Optional[threading.Thread] = None
        self.capture: Any = None; self.capture_lock = threading.Lock(); self.closed = False
        self.module_enabled = {"Person Detection": True, "Multi-Person Tracking": True, "Interaction Analysis": True, "Rules Engine": True}
        self.source_fps, self.frames_received, self.capture_started = 0.0, 0, 0.0
        self._build_control(); self._build_preview(); self._build_data()
        self.root.after(self.config.gui_update_interval_ms, self._refresh)

    def _window(self, title: str, geometry: str):
        window = self.tk.Toplevel(self.root); window.title(title); window.geometry(geometry)
        window.protocol("WM_DELETE_WINDOW", window.withdraw); return window

    def _build_control(self) -> None:
        tk, ttk = self.tk, self.ttk; self.control = self._window("AI CAMERA CONTROL", "540x430")
        # The control window is the application owner; closing it shuts down safely.
        self.control.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(self.control, padding=10); frame.pack(fill="both", expand=True)
        self.source_kind = tk.StringVar(value="IP Webcam"); self.source_value = tk.StringVar(value="http://192.168.1.10:8080/video")
        self.status = tk.StringVar(value="Ready — M5 consumes M4-compatible features")
        ttk.Label(frame, text="AI CAMERA CONTROL", font=("Arial", 13, "bold")).pack(anchor="w")
        row = ttk.Frame(frame); row.pack(fill="x", pady=5)
        ttk.Label(row, text="Input:").pack(side="left"); ttk.Combobox(row, textvariable=self.source_kind, values=("IP Webcam", "Video File"), state="readonly", width=14).pack(side="left", padx=4)
        ttk.Entry(row, textvariable=self.source_value, width=36).pack(side="left", padx=4)
        ttk.Button(row, text="Browse", command=self._choose_file).pack(side="left")
        buttons = ttk.Frame(frame); buttons.pack(fill="x", pady=5)
        for text, command in (("Start", self.start), ("Pause", self.pause), ("Resume", self.resume), ("Stop", self.stop)):
            ttk.Button(buttons, text=text, command=command).pack(side="left", padx=3)
        mods = ttk.LabelFrame(frame, text="AI modules", padding=6); mods.pack(fill="x", pady=5)
        self.modules = {n: tk.BooleanVar(value=True) for n in self.module_enabled}
        for name, var in self.modules.items():
            ttk.Checkbutton(mods, text=name, variable=var, command=lambda n=name, v=var: self.module_enabled.__setitem__(n, bool(v.get()))).pack(side="left", padx=4)
        self.metrics = tk.StringVar(value="Source FPS: —   Processing FPS: —   Device: CPU/CUDA determined by upstream")
        ttk.Label(frame, textvariable=self.metrics).pack(anchor="w", pady=(8, 2)); ttk.Label(frame, textvariable=self.status, wraplength=500).pack(anchor="w")

    def _build_preview(self) -> None:
        ttk = self.ttk; self.preview_window = self._window("CAMERA PREVIEW", "900x650")
        frame = ttk.Frame(self.preview_window, padding=8); frame.pack(fill="both", expand=True)
        self.preview_status = self.tk.StringVar(value="Waiting for camera frames.")
        self.preview_label = ttk.Label(frame, textvariable=self.preview_status, anchor="center"); self.preview_label.pack(fill="both", expand=True)
        self.preview_footer = self.tk.StringVar(value="Highest risk: NORMAL")
        ttk.Label(frame, textvariable=self.preview_footer).pack(anchor="w", pady=4)

    def _build_data(self) -> None:
        tk, ttk = self.tk, self.ttk; self.data_window = self._window("SUSPICIOUS DATA", "650x680")
        frame = ttk.Frame(self.data_window, padding=10); frame.pack(fill="both", expand=True)
        self.summary = tk.StringVar(value="Waiting for M4 interaction data…"); ttk.Label(frame, textvariable=self.summary, font=("Arial", 12, "bold"), justify="left").pack(anchor="w")
        self.bar = ttk.Progressbar(frame, length=450, maximum=100); self.bar.pack(fill="x", pady=8)
        self.signal_text = tk.Text(frame, height=16, wrap="word", state="disabled"); self.signal_text.pack(fill="both", expand=True)
        ttk.Label(frame, text="Recent Events").pack(anchor="w", pady=(8, 1))
        self.events = ttk.Treeview(frame, columns=("Time", "Pair", "Score", "State"), show="headings", height=6)
        for col, width in (("Time", 110), ("Pair", 110), ("Score", 80), ("State", 120)):
            self.events.heading(col, text=col); self.events.column(col, width=width, anchor="center")
        self.events.pack(fill="x")

    def _choose_file(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All files", "*.*")])
        if path:
            self.source_kind.set("Video File"); self.source_value.set(path)

    def start(self) -> None:
        if self.capture_worker and self.capture_worker.is_alive():
            self.paused.clear(); self.status.set("RECEIVING FRAMES"); return
        source = normalize_source(self.source_value.get(), self.source_kind.get())
        if not source or (self.source_kind.get() == "Video File" and not Path(source).is_file()):
            self.status.set("ERROR: choose a readable video file or an IP Webcam URL."); return
        self.source_value.set(source); self.stop_event.clear(); self.paused.clear(); self.engine.reset(); self.capture_started = time.monotonic(); self.frames_received = 0
        self.capture_worker = threading.Thread(target=self._capture_loop, args=(source, self.source_kind.get()), name="m5-camera-capture", daemon=True)
        self.capture_worker.start()
        if not self.worker or not self.worker.is_alive():
            self.worker = threading.Thread(target=self._work, name="m5-rules-worker", daemon=True); self.worker.start()
        self.status.set(f"CONNECTING: {source}")

    def _capture_loop(self, source: str, source_kind: str) -> None:
        """Dedicated latest-frame reader. No Tk calls occur on this thread."""
        try:
            import cv2
        except Exception as exc:
            self.diagnostics.put({"state": "ERROR", "message": f"OpenCV unavailable: {exc}"}); return
        failures = 0
        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(source)
            try: cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.capture_buffer_size)
            except Exception: pass
            with self.capture_lock: self.capture = cap
            if not cap.isOpened():
                cap.release(); self.diagnostics.put({"state": "ERROR", "message": f"Unable to open IP Webcam/video stream: {source}"})
                if self.stop_event.wait(self.config.reconnect_delay_seconds): break
                continue
            self.source_fps = _finite(cap.get(cv2.CAP_PROP_FPS)) or 0.0; failures = 0
            self.diagnostics.put({"state": "CONNECTED", "message": f"Opened {source}"})
            while not self.stop_event.is_set():
                if self.paused.is_set():
                    if self.stop_event.wait(.05): break
                    continue
                ok, frame = cap.read()
                if not ok or frame is None or getattr(frame, "ndim", 0) < 2:
                    failures += 1
                    if source_kind == "Video File":
                        self.diagnostics.put({"state": "STOPPED", "message": "Video file ended."}); self.stop_event.set(); break
                    self.diagnostics.put({"state": "NO FRAME", "message": f"Stream opened but no frame received ({failures}); reconnecting."}); break
                now = time.monotonic(); self.frames.put((frame, now)); self.frames_received += 1
                self.diagnostics.put({"state": "RECEIVING FRAMES", "message": f"{frame.shape[1]}×{frame.shape[0]} BGR frame", "last_frame": now})
            cap.release()
            with self.capture_lock: self.capture = None
            if not self.stop_event.is_set() and self.stop_event.wait(self.config.reconnect_delay_seconds): break

    def _work(self) -> None:
        last_frame_at = -1.0; models_loaded = False
        while not self.stop_event.is_set():
            if self.paused.is_set():
                self.stop_event.wait(.05)
                continue
            try:
                snapshot = self.frames.get()
                upstream = self.feature_source() if self.feature_source and self.module_enabled["Interaction Analysis"] else None
                if snapshot and snapshot[1] != last_frame_at and self.module_enabled["Person Detection"]:
                    frame, frame_at = snapshot; last_frame_at = frame_at
                    if not models_loaded:
                        self.detector.load_models(); models_loaded = True
                    people = self.detector.infer(frame, pose=True, enabled=True)
                    if self.module_enabled["Multi-Person Tracking"]: people = self.tracker.update(people, frame_at)
                    if self.module_enabled["Interaction Analysis"]: upstream = self.interactions.process(people, frame_at)
                    try:
                        import cv2
                        drawn = frame.copy()
                        for p in people:
                            x1,y1,x2,y2=p["bbox"]; cv2.rectangle(drawn,(x1,y1),(x2,y2),(0,220,0),2); cv2.putText(drawn,f"ID {p.get('person_id','?')}",(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,0),1)
                        self.processed_frames.put((drawn, frame_at))
                    except Exception as exc: LOGGER.warning("M5 overlay unavailable: %s",exc)
                if upstream and self.module_enabled["Rules Engine"]:
                    self.results.put(self.engine.evaluate_batch(upstream))
                    self.diagnostics.put({"state":"AI PROCESSING","message":f"Detection {len(people) if 'people' in locals() else 0} | Interaction {len(upstream.get('pairs',()))} pairs | Rules active"})
            except Exception as exc:
                LOGGER.exception("M5 feature worker failure: %s", exc)
                self.diagnostics.put({"state": "AI ERROR", "message": f"Rules/input error: {exc}"})
            self.stop_event.wait(.04)

    def pause(self) -> None: self.paused.set(); self.status.set("PAUSED — capture held")
    def resume(self) -> None: self.paused.clear(); self.status.set("RECEIVING FRAMES")
    def stop(self) -> None:
        self.stop_event.set()
        with self.capture_lock:
            if self.capture is not None: self.capture.release()
        self.status.set("STOPPED")

    def _refresh(self) -> None:
        if self.closed: return
        diagnostic = self.diagnostics.get()
        if diagnostic:
            self.status.set(f"{diagnostic['state']}: {diagnostic['message']}")
        snapshot = self.processed_frames.get() or self.frames.get()
        if snapshot:
            frame, received_at = snapshot
            try:
                import cv2
                from PIL import Image, ImageTk
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if getattr(frame, "ndim", 0) == 3 else frame
                image = Image.fromarray(rgb); image.thumbnail((860, 570))
                photo = ImageTk.PhotoImage(image=image)
                self.preview_label.configure(image=photo, text=""); self.preview_label.image = photo
                elapsed = max(time.monotonic() - self.capture_started, .001) if self.capture_started else 0.0
                self.metrics.set(f"Source FPS: {self.source_fps:.1f}   Received: {self.frames_received}   Display: live   Frame age: {time.monotonic()-received_at:.2f}s")
            except Exception as exc:
                self.preview_status.set(f"PREVIEW ERROR: frame received but display failed: {exc}")
        payload = self.results.get()
        if payload:
            best = payload.get("highest_risk")
            if best:
                score, pair, state = best["suspiciousness_score"], best["pair_ids"], best["risk_state"]
                self.summary.set(f"Person #{pair[0]} ↔ Person #{pair[1]}\nSuspiciousness: {score:.0f} / 100   State: {state}")
                self.bar["value"] = score; self.preview_footer.set(f"Highest risk: {state} ({score:.0f}/100) — Human review when elevated")
                lines = [f"Episode: {best['episode_id']}   Duration: {best['interaction_duration_seconds']:.1f}s", ""]
                lines.extend(f"{name.replace('_', ' ').title()}: {value * 100:.0f}%" for name, value in best["normalized_signals"].items() if best["signal_available"].get(name, True))
                lines += ["", "Reasons: " + ("; ".join(best["reasons"]) or "No elevated rule currently active."),
                          f"Alert: {'generated' if best['alert_generated'] else 'suppressed/none'}; cooldown {best['cooldown_remaining_seconds']:.1f}s"]
                self.signal_text.configure(state="normal"); self.signal_text.delete("1.0", "end"); self.signal_text.insert("1.0", "\n".join(lines)); self.signal_text.configure(state="disabled")
            for iid in self.events.get_children(): self.events.delete(iid)
            for event in payload.get("recent_events", [])[:self.config.event_history_size]:
                self.events.insert("", "end", values=(f"{event['timestamp']:.1f}", f"#{event['pair_ids'][0]} ↔ #{event['pair_ids'][1]}", f"{event['score']:.0f}", event["state"]))
        try: self.root.after(self.config.gui_update_interval_ms, self._refresh)
        except self.tk.TclError: pass

    def close(self) -> None:
        if self.closed: return
        self.closed = True; self.stop_event.set()
        with self.capture_lock:
            if self.capture is not None: self.capture.release()
        for worker in (self.capture_worker, self.worker):
            if worker and worker.is_alive() and worker is not threading.current_thread(): worker.join(timeout=2)
        for window in (getattr(self, "control", None), getattr(self, "preview_window", None), getattr(self, "data_window", None)):
            try:
                if window and window.winfo_exists(): window.destroy()
            except self.tk.TclError: pass
        self.root.destroy()

    def run(self) -> None: self.root.mainloop()


def _synthetic_pair(t: float, distance: float, **values: Any) -> Dict[str, Any]:
    result = {"person_a": 17, "person_b": 22, "timestamp": t, "normalized_distance": distance,
              "proximity_level": "VERY_CLOSE" if distance <= .85 else "CLOSE" if distance <= 1.65 else "MODERATE" if distance <= 3 else "FAR",
              "following_score": 0., "approach_score": 0., "a_speed": 0., "b_speed": 0., "interaction_score": .5}
    result.update(values); return result


def run_synthetic_tests() -> None:
    """Representative deterministic smoke tests; no GUI, models, or M4 imports."""
    engine = RulesEngineM5(M5Config(alert_cooldown_seconds=2., stale_pair_timeout_seconds=3.))
    def stream(distance: float, seconds: int = 10, **kw: Any) -> Dict[str, Any]:
        last = None
        for i in range(seconds * 5): last = engine.evaluate(_synthetic_pair(i / 5, distance, **kw))
        assert last is not None; return last
    assert stream(4.0)["risk_state"] in ("NORMAL", "LOW")                 # normal distant / high movement at distance
    engine.reset(); assert stream(4.0, 8, a_speed=300)["risk_state"] != "HIGH_RISK"
    engine.reset(); assert stream(1.0, 1)["risk_state"] != "HIGH_RISK"       # brief close
    engine.reset(); assert stream(1.0, 12)["suspiciousness_score"] > 15       # prolonged close
    engine.reset(); assert stream(1.0, 1, following_score=.98)["risk_state"] != "HIGH_RISK"
    engine.reset(); assert stream(1.0, 12, following_score=.98)["suspiciousness_score"] > 30  # sustained following
    engine.reset(); assert stream(1.0, 12, following_score=.95, approach_score=.95, a_speed=190, hand_near_body_score=.9, body_overlap=.65)["risk_state"] == "HIGH_RISK"
    engine.reset(); assert stream(1.0, 3, hand_near_body_score=.99)["risk_state"] != "HIGH_RISK"  # hand alone
    # Reversed IDs canonicalize, supplied group input is independent, and cooldown deduplicates.
    engine.reset(); first = stream(1.0, 12, following_score=.95, approach_score=.95, a_speed=190, hand_near_body_score=.9, body_overlap=.65)
    later = engine.evaluate(_synthetic_pair(12.1, 1.0, following_score=.95, approach_score=.95, a_speed=190, hand_near_body_score=.9, body_overlap=.65))
    assert first["alert_generated"] or engine.recent_events
    assert later and later["alert_generated"] is False
    group = engine.evaluate({**_synthetic_pair(13., 1.0), "person_a": 99, "person_b": 98, "nearby_person_count": 3})
    assert group and group["signal_available"]["group_surrounding"]
    malformed = engine.evaluate({"person_a": 2, "person_b": 1, "normalized_distance": float("nan"), "following_score": None}, 100)
    assert malformed and malformed["pair_ids"] == (1, 2) and 0 <= malformed["suspiciousness_score"] <= 100
    engine.cleanup(200); assert not engine._states                              # missing detections / episode termination / stale cleanup
    print("M5 synthetic tests passed")


def _demo_source() -> Callable[[], Mapping[str, Any]]:
    started = time.monotonic()
    def source() -> Mapping[str, Any]:
        elapsed = time.monotonic() - started
        distance = max(.7, 3.2 - elapsed * .18)
        return {"timestamp": elapsed, "pairs": [_synthetic_pair(elapsed, distance, following_score=.85 if elapsed > 4 else .1, approach_score=.9 if 3 < elapsed < 8 else 0., a_speed=130 if elapsed > 5 else 15, hand_near_body_score=.7 if elapsed > 8 else 0.)]}
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description="M5 safety-risk rules engine")
    parser.add_argument("--test", action="store_true", help="run deterministic synthetic validation")
    parser.add_argument("--demo", action="store_true", help="run the three-window synthetic demonstration")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.test: run_synthetic_tests(); return
    app = M5Application(M5Config(debug=args.debug), _demo_source() if args.demo else None)
    if args.demo: app.start()
    app.run()


if __name__ == "__main__":
    main()
