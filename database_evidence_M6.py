"""Module 6: protected-target behavior analysis and evidence integration.

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
import csv
import json
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
from datetime import datetime, timezone
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


@dataclass
class M6Config:
    """Evidence/database settings. Values are intentionally independent of M5 tuning."""
    evidence_directory: str = "evidence"
    buffer_directory: str = ".evidence_buffer"
    pre_event_seconds: float = 300.0
    post_event_seconds: float = 300.0
    segment_seconds: float = 20.0
    recorder_queue_size: int = 3
    finalizer_queue_size: int = 8
    event_history_size: int = 100
    max_event_seconds: float = 1800.0
    db_queue_size: int = 100
    mysql_host: str = field(default_factory=lambda: os.getenv("MYSQL_HOST", "127.0.0.1"))
    mysql_port: int = field(default_factory=lambda: int(os.getenv("MYSQL_PORT", "3306")))
    mysql_database: str = field(default_factory=lambda: os.getenv("MYSQL_DATABASE", "women_safety"))
    mysql_user: str = field(default_factory=lambda: os.getenv("MYSQL_USER", "saniya"))
    mysql_password: str = field(default_factory=lambda: os.getenv("MYSQL_PASSWORD", ""))


@dataclass
class PersonMetadataConfig:
    """Manual, non-sensitive metadata settings; no video classification is used."""
    enabled: bool = True
    cache_timeout_seconds: float = 3.0
    max_note_length: int = 120


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

    def update_parameter(self, section: str, name: str, value: float) -> bool:
        """Atomically update the same configuration object used by evaluation."""
        value = _finite(value)
        if value is None: return False
        with self._lock:
            cfg = self.config
            if section == "weights":
                if name not in cfg.weights or not 0 <= value <= 100: return False
                cfg.weights[name] = value
            elif section in ("enter_thresholds", "exit_thresholds"):
                if name not in getattr(cfg, section) or not 0 <= value <= 100: return False
                trial = dict(getattr(cfg, section)); trial[name] = value
                if list(trial.values()) != sorted(trial.values()): return False
                if section == "exit_thresholds" and any(trial[k] >= cfg.enter_thresholds[k] for k in trial): return False
                setattr(cfg, section, trial)
            else:
                ranges = {"ema_alpha":(0.,1.),"rapid_approach_score":(0.,1.),"escalation_distance_delta":(0.,5.),"high_speed":(0.,1000.),"high_speed_full":(1.,1500.),"close_persistence_seconds":(0.,120.),"duration_full_seconds":(.1,600.),"following_persistence_seconds":(0.,120.),"movement_persistence_seconds":(0.,120.),"hand_persistence_seconds":(0.,120.),"overlap_persistence_seconds":(0.,120.),"episode_gap_seconds":(0.,600.),"alert_cooldown_seconds":(0.,3600.)}
                lo, hi = ranges.get(name, (0., 1000.))
                if not hasattr(cfg, name) or not lo <= value <= hi: return False
                setattr(cfg, name, value)
            return True

    def configuration(self) -> Dict[str, Any]:
        with self._lock: return copy.deepcopy(asdict(self.config))

    def load_configuration(self, data: Mapping[str, Any]) -> bool:
        """Validate a versioned JSON payload before replacing live values."""
        try:
            values = data["config"] if "config" in data else data
            with self._lock:
                before = copy.deepcopy(self.config)
                for section in ("weights", "enter_thresholds", "exit_thresholds"):
                    if section in values:
                        for key, val in values[section].items():
                            if not self.update_parameter(section, key, val): self.config.__dict__.update(before.__dict__); return False
                for key, val in values.items():
                    if key not in ("weights", "enter_thresholds", "exit_thresholds") and hasattr(self.config,key) and isinstance(val,(int,float)):
                        if not self.update_parameter("scalar",key,val): self.config.__dict__.update(before.__dict__); return False
            return True
        except (TypeError, KeyError): return False

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


class PersonMetadataManager:
    """Thread-safe bounded store for operator-entered, neutral track metadata."""
    def __init__(self, config: Optional[PersonMetadataConfig] = None) -> None:
        self.config = config or PersonMetadataConfig()
        self._results: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def submit_latest(self, frame: Any, people: Sequence[Mapping[str, Any]], timestamp: float) -> None:
        """Keep the existing call boundary without inspecting frames."""
        with self._lock:
            for person in people:
                track_id = person.get("person_id")
                if track_id is not None and int(track_id) not in self._results:
                    self._results[int(track_id)] = {"track_id": int(track_id), "metadata": "Not Classified",
                                                    "source": "Unavailable", "status": "Unavailable", "timestamp": timestamp}
            self._prune(timestamp)

    def set_manual_metadata(self, track_id: Any, note: str, timestamp: Optional[float] = None) -> bool:
        try:
            track_id = int(track_id); note = str(note).strip()[:self.config.max_note_length]
        except (TypeError, ValueError):
            return False
        if not note:
            return False
        now = time.time() if timestamp is None else timestamp
        with self._lock:
            self._results[track_id] = {"track_id": track_id, "metadata": note, "source": "Manual",
                                       "status": "Operator supplied", "timestamp": now}
            self._prune(now)
        return True

    def snapshot(self, timestamp: Optional[float] = None) -> Dict[int, Dict[str, Any]]:
        with self._lock:
            self._prune(time.time() if timestamp is None else timestamp)
            return copy.deepcopy(self._results)

    def _prune(self, timestamp: float) -> None:
        for track_id, result in list(self._results.items()):
            if timestamp - float(result["timestamp"]) > self.config.cache_timeout_seconds:
                self._results.pop(track_id, None)


@dataclass
class ProtectedTargetConfig:
    cache_timeout_seconds: float = 3.0


class ProtectedTargetManager:
    """Operator-selected protected Track IDs and a read-only pair view."""
    def __init__(self, config: Optional[ProtectedTargetConfig] = None) -> None:
        self.config = config or ProtectedTargetConfig()
        self._targets: Dict[int, float] = {}
        self._active_tracks: set[int] = set()
        self._lock = threading.RLock()

    def set_target(self, track_id: Any, timestamp: Optional[float] = None) -> bool:
        try: track_id = int(track_id)
        except (TypeError, ValueError): return False
        now = time.monotonic() if timestamp is None else timestamp
        with self._lock:
            if track_id not in self._active_tracks: return False
            self._targets[track_id] = now
        return True

    def clear_target(self, track_id: Any) -> None:
        try: track_id = int(track_id)
        except (TypeError, ValueError): return
        with self._lock: self._targets.pop(track_id, None)

    def update_tracks(self, people: Sequence[Mapping[str, Any]], timestamp: float) -> None:
        active = {int(person["person_id"]) for person in people if person.get("person_id") is not None}
        with self._lock:
            self._active_tracks = active
            self._targets = {track_id: timestamp if track_id in active else seen for track_id, seen in self._targets.items()
                             if track_id in active or timestamp - seen <= self.config.cache_timeout_seconds}

    def is_target(self, track_id: Any) -> bool:
        try: track_id = int(track_id)
        except (TypeError, ValueError): return False
        with self._lock: return track_id in self._targets

    def snapshot(self) -> List[int]:
        with self._lock: return sorted(self._targets)

    def protected_pairs(self, pairs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        with self._lock: targets = set(self._targets)
        output = []
        for pair in pairs:
            a, b = int(pair["person_a"]), int(pair["person_b"])
            for target_id, other_id, moving_away in ((a, b, pair.get("a_moving_away", False)), (b, a, pair.get("b_moving_away", False))):
                if target_id in targets:
                    output.append({**pair, "protected_target_id": target_id, "other_person_id": other_id,
                                   "target_status": "Protected Target", "target_moving_away": moving_away})
        return output


@dataclass
class SegmentInfo:
    path: Path; start: float; end: float; frames: int; fps: float; resolution: Tuple[int, int]; session_id: str


@dataclass
class ManagedEvent:
    event_id: str; episode_id: str; pair_ids: PairKey; camera_id: str; source_type: str; source: str; session_id: str
    started_at: float; last_active_at: float; finalize_after: float; trigger: Dict[str, Any]
    peak_score: float; final_score: float; peak_state: str; rules: set[str] = field(default_factory=set)
    pre_segments: List[SegmentInfo] = field(default_factory=list); status: str = "RECORDING"


class RollingEvidenceRecorder:
    """One bounded recording worker writes short segments; no raw-frame history is retained."""
    def __init__(self, cfg: M6Config, session_id: str, on_segment: Callable[[SegmentInfo], None]) -> None:
        self.cfg,self.session_id,self.on_segment=cfg,session_id,on_segment; self.frames: "queue.Queue[Tuple[Any,float]]"=queue.Queue(cfg.recorder_queue_size)
        self.stop_event=threading.Event(); self.thread: Optional[threading.Thread]=None; self.status=LatestSlot()
    def start(self) -> None:
        Path(self.cfg.buffer_directory).mkdir(parents=True,exist_ok=True); self.stop_event.clear(); self.thread=threading.Thread(target=self._run,name="m6-evidence-recorder",daemon=True); self.thread.start()
    def submit(self, frame: Any, timestamp: float) -> None:
        try: self.frames.put_nowait((frame,timestamp))
        except queue.Full:
            try: self.frames.get_nowait()
            except queue.Empty: pass
            try: self.frames.put_nowait((frame,timestamp))
            except queue.Full: pass
    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive(): self.thread.join(timeout=5)
    def _run(self) -> None:
        writer=None; first=last=0.; count=0; path: Optional[Path]=None; fps=20.; size=(0,0)
        try:
            import cv2
            while not self.stop_event.is_set() or not self.frames.empty():
                try: frame,ts=self.frames.get(timeout=.2)
                except queue.Empty: continue
                if writer is None:
                    h,w=frame.shape[:2]; size=(w,h); first=ts; count=0; path=Path(self.cfg.buffer_directory)/f"{self.session_id}_{int(ts*1000)}.mp4"
                    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,size)
                    if not writer.isOpened(): raise RuntimeError(f"Cannot create evidence segment: {path}")
                writer.write(frame); last=ts; count+=1
                if last-first>=self.cfg.segment_seconds:
                    writer.release(); self.on_segment(SegmentInfo(path,first,last,count,fps,size,self.session_id)); writer=None
            if writer is not None:
                writer.release(); self.on_segment(SegmentInfo(path,first,last,count,fps,size,self.session_id))
        except Exception as exc: LOGGER.exception("[M6] recorder failure: %s",exc); self.status.put({"state":"ERROR","message":str(exc)})


class DatabaseWorker:
    """Owns an optional MySQL connection; database failures never block evidence or AI."""
    def __init__(self,cfg:M6Config) -> None: self.cfg=cfg; self.jobs: "queue.Queue[Dict[str,Any]]"=queue.Queue(cfg.db_queue_size); self.stop_event=threading.Event(); self.thread: Optional[threading.Thread]=None; self.status=LatestSlot()
    def start(self)->None: self.thread=threading.Thread(target=self._run,name="m6-mysql-worker",daemon=True); self.thread.start()
    def submit(self,job:Dict[str,Any])->None:
        try:self.jobs.put_nowait(job)
        except queue.Full: self.status.put({"state":"DB QUEUE FULL","message":"event database job dropped; local evidence is safe"})
    def stop(self)->None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive(): self.thread.join(timeout=5)
    def _run(self)->None:
        conn=None
        while not self.stop_event.is_set() or not self.jobs.empty():
            try: job=self.jobs.get(timeout=.25)
            except queue.Empty: continue
            try:
                if conn is None:
                    import mysql.connector
                    conn=mysql.connector.connect(host=self.cfg.mysql_host,port=self.cfg.mysql_port,database=self.cfg.mysql_database,user=self.cfg.mysql_user,password=self.cfg.mysql_password,connection_timeout=5)
                    cur=conn.cursor(); cur.execute("CREATE TABLE IF NOT EXISTS risk_events (event_id VARCHAR(80) PRIMARY KEY, camera_id VARCHAR(128), event_start DOUBLE, event_end DOUBLE, person_1_id INT, person_2_id INT, risk_score DOUBLE, risk_state VARCHAR(32), triggered_rules TEXT, evidence_video_path TEXT, metadata_csv_path TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"); conn.commit()
                cur=conn.cursor(); cur.execute("INSERT INTO risk_events (event_id,camera_id,event_start,event_end,person_1_id,person_2_id,risk_score,risk_state,triggered_rules,evidence_video_path,metadata_csv_path) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE event_end=VALUES(event_end),risk_score=VALUES(risk_score),risk_state=VALUES(risk_state),evidence_video_path=VALUES(evidence_video_path),metadata_csv_path=VALUES(metadata_csv_path)",(job["event_id"],job["camera_id"],job["event_start"],job["event_end"],job["person_1_id"],job["person_2_id"],job["risk_score"],job["risk_state"],job["triggered_rules"],job["video_path"],job["csv_path"])); conn.commit(); self.status.put({"state":"DB SAVED","message":job["event_id"]})
            except Exception as exc:
                LOGGER.warning("[M6] database job failed: %s",exc); self.status.put({"state":"DB UNAVAILABLE","message":str(exc)})
                try:
                    if conn: conn.close()
                finally: conn=None
        if conn:
            try: conn.close()
            except Exception: pass


class EventManager:
    """Maps M5 episode/risk results to event evidence without a second risk decision layer."""
    def __init__(self,cfg:M6Config,db:DatabaseWorker) -> None:
        self.cfg,self.db=cfg,db; self.lock=threading.RLock(); self.segments: Deque[SegmentInfo]=deque(); self.active: Dict[str,ManagedEvent]={}; self.history:Deque[Dict[str,Any]]=deque(maxlen=cfg.event_history_size); self.finalize: "queue.Queue[ManagedEvent]"=queue.Queue(cfg.finalizer_queue_size); self.stop_event=threading.Event(); self.thread=threading.Thread(target=self._finalizer,name="m6-evidence-finalizer",daemon=True); self.counter=0; self.status=LatestSlot()
    def start(self)->None: Path(self.cfg.evidence_directory).mkdir(parents=True,exist_ok=True); self.thread.start()
    def add_segment(self,segment:SegmentInfo)->None:
        with self.lock:
            self.segments.append(segment)
            cutoff=segment.end-self.cfg.pre_event_seconds
            while self.segments and self.segments[0].end<cutoff:
                old=self.segments.popleft()
                protected=any(old.path in {s.path for s in event.pre_segments} for event in self.active.values())
                if protected:
                    self.segments.appendleft(old); break
                try: old.path.unlink(missing_ok=True)
                except OSError: LOGGER.warning("[M6] could not remove old segment %s",old.path)
    def observe(self,result:Mapping[str,Any],camera_id:str,source_type:str,source:str,session_id:str)->None:
        if result.get("risk_state") not in ("SUSPICIOUS","HIGH_RISK"): return
        now=float(result["timestamp"]); key=str(result.get("episode_id") or result["pair_ids"])
        with self.lock:
            event=self.active.get(key)
            if event is None:
                self.counter+=1; eid=f"EVT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.counter:04d}"; start=now-self.cfg.pre_event_seconds
                pre=[s for s in self.segments if s.end>=start and s.session_id==session_id]
                event=ManagedEvent(eid,key,tuple(result["pair_ids"]),camera_id,source_type,source,session_id,now,now,now+self.cfg.post_event_seconds,dict(result),result["suspiciousness_score"],result["suspiciousness_score"],result["risk_state"],set(result.get("triggered_rule_ids",())),pre); self.active[key]=event; self.status.put({"state":"EVENT RECORDING","message":eid})
            event.last_active_at=now; event.finalize_after=min(now+self.cfg.post_event_seconds,event.started_at+self.cfg.max_event_seconds); event.peak_score=max(event.peak_score,result["suspiciousness_score"]); event.final_score=result["suspiciousness_score"]; event.peak_state="HIGH_RISK" if result["risk_state"]=="HIGH_RISK" else event.peak_state; event.rules.update(result.get("triggered_rule_ids",()))
    def tick(self,now:float)->None:
        with self.lock:
            for key,event in list(self.active.items()):
                if now>=event.finalize_after:
                    self.active.pop(key); event.status="FINALIZING"
                    try:self.finalize.put_nowait(event)
                    except queue.Full: LOGGER.error("[M6] finalizer queue full for %s",event.event_id)
    def current_event(self) -> Optional[ManagedEvent]:
        with self.lock: return next(iter(self.active.values()),None)
    def stop(self)->None:
        self.stop_event.set()
        with self.lock:
            for e in self.active.values():
                try:self.finalize.put_nowait(e)
                except queue.Full: pass
            self.active.clear()
        if self.thread.is_alive(): self.thread.join(timeout=15)
    def _finalizer(self)->None:
        while not self.stop_event.is_set() or not self.finalize.empty():
            try:event=self.finalize.get(timeout=.25)
            except queue.Empty:continue
            try:self._write_evidence(event)
            except Exception as exc: LOGGER.exception("[M6] evidence finalization failed: %s",exc); self.status.put({"state":"EVIDENCE ERROR","message":str(exc)})
    def _write_evidence(self,event:ManagedEvent)->None:
        import cv2
        with self.lock: segments=[s for s in self.segments if s.session_id==event.session_id and s.end>=event.pre_segments[0].start] if event.pre_segments else list(self.segments)
        segments=list({str(s.path):s for s in event.pre_segments+segments}.values()); folder=Path(self.cfg.evidence_directory)/event.event_id; folder.mkdir(parents=True,exist_ok=False); video=folder/f"{event.event_id}.mp4"; writer=None; frames=0; first=None; last=None
        for seg in segments:
            cap=cv2.VideoCapture(str(seg.path))
            while True:
                ok,frame=cap.read()
                if not ok:break
                if writer is None:
                    h,w=frame.shape[:2]; writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),seg.fps or 20.,(w,h)); first=seg.start
                writer.write(frame); frames+=1
            cap.release(); last=seg.end
        if writer: writer.release()
        actual_pre=max(0.,event.started_at-(first or event.started_at)); duration=max(0.,(last or event.started_at)-(first or event.started_at)); csv_path=folder/f"{event.event_id}.csv"; r=event.trigger; signals=r.get("normalized_signals",{})
        row={"event_id":event.event_id,"camera_id":event.camera_id,"source_type":event.source_type,"source":event.source,"session_id":event.session_id,"event_start_time":event.started_at,"event_end_time":event.last_active_at,"evidence_start_time":first or "N/A","evidence_end_time":last or "N/A","pre_event_requested_seconds":self.cfg.pre_event_seconds,"pre_event_actual_seconds":actual_pre,"post_event_requested_seconds":self.cfg.post_event_seconds,"post_event_actual_seconds":max(0.,(last or event.last_active_at)-event.last_active_at),"video_duration_seconds":duration,"video_file":str(video),"risk_score_at_trigger":r.get("suspiciousness_score","N/A"),"peak_risk_score":event.peak_score,"final_risk_score":event.final_score,"risk_state_at_trigger":r.get("risk_state","N/A"),"peak_risk_state":event.peak_state,"person_1_id":event.pair_ids[0],"person_2_id":event.pair_ids[1],"triggered_rules":";".join(sorted(event.rules)),"top_contributors":";".join(k for k,_ in sorted(r.get("rule_contributions",{}).items(),key=lambda x:x[1],reverse=True)[:3]),"configuration_snapshot":json.dumps(r.get("rule_weights",{}))}; row.update({k:signals.get(k,"N/A") for k in ("persistent_proximity","following","rapid_approach","movement_context","body_overlap","hand_proximity","facing_context","repeated_approach","group_surrounding")})
        with csv_path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerow(row)
        event.status="SAVED"; summary={"event_id":event.event_id,"score":event.peak_score,"state":event.peak_state,"status":"SAVED","video_path":str(video),"csv_path":str(csv_path)}; self.history.appendleft(summary); self.db.submit({"event_id":event.event_id,"camera_id":event.camera_id,"event_start":event.started_at,"event_end":event.last_active_at,"person_1_id":event.pair_ids[0],"person_2_id":event.pair_ids[1],"risk_score":event.peak_score,"risk_state":event.peak_state,"triggered_rules":";".join(sorted(event.rules)),"video_path":str(video),"csv_path":str(csv_path)}); self.status.put({"state":"EVIDENCE SAVED","message":event.event_id})


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
            key=(ia,ib); s=self.states.setdefault(key,{"first":ts,"last":ts,"dist":deque(maxlen=10),"close":0.,"follow":0.,"approach":0.,"hand":0.,"a_away":0.,"b_away":0.}); dt=max(0.,ts-s["last"]); prev=s["dist"][-1][1] if s["dist"] else None; s["dist"].append((ts,dist)); s["last"]=ts
            level="VERY_CLOSE" if dist<=M4_VERY_CLOSE else "CLOSE" if dist<=M4_CLOSE else "MODERATE"; s["close"]+=dt if level in ("VERY_CLOSE","CLOSE") else 0.
            ds=[x[1] for x in s["dist"]]; approach=0.; approaching=False; separating=False
            if len(ds)>=3 and s["dist"][-1][0]-s["dist"][0][0]>=.6:
                tt=[x[0] for x in s["dist"]]; mt,md=statistics.mean(tt),statistics.mean(ds); den=sum((x-mt)**2 for x in tt); slope=sum((x-mt)*(y-md) for x,y in zip(tt,ds))/den if den else 0.; consistency=sum(y<x for x,y in zip(ds,ds[1:]))/(len(ds)-1); approaching=slope<-.08 and consistency>=.65; separating=slope>.08 and consistency<=.35; approach=min(1.,abs(slope)/.32)*(consistency if slope<0 else 0.)
            va,vb=motions[ia],motions[ib]; ua,ub=_m4_unit((va["velocity_x"],va["velocity_y"])),_m4_unit((vb["velocity_x"],vb["velocity_y"])); direction=max(0.,_m4_dot(ua,ub)) if ua and ub else 0.; velocity=1-min(1.,abs(va["speed"]-vb["speed"])/max(va["speed"],vb["speed"],1.)); count=min(len(self.positions[ia]),len(self.positions[ib]),12); traj=0.
            a_away = _m4_dot((va["velocity_x"], va["velocity_y"]), (ca[0]-cb[0], ca[1]-cb[1])) > 8.0
            b_away = _m4_dot((vb["velocity_x"], vb["velocity_y"]), (cb[0]-ca[0], cb[1]-ca[1])) > 8.0
            s["a_away"] = s["a_away"] + dt if a_away else 0.0; s["b_away"] = s["b_away"] + dt if b_away else 0.0
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
            output.append({"person_a":ia,"person_b":ib,"pixel_distance":math.dist(ca,cb),"normalized_distance":dist,"previous_distance":prev,"distance_delta":None if prev is None else dist-prev,"proximity_level":level,"approaching":approaching,"separating":separating,"distance_trend":"DECREASING" if approaching else "INCREASING" if separating else "STABLE","approach_score":approach,"following_score":following,"following":following_on,"a_speed":va["speed"],"b_speed":vb["speed"],"a_moving_away":s["a_away"] >= .6,"b_moving_away":s["b_away"] >= .6,"facing_score":face,"facing_each_other":None if face is None else face>=.60,"facing_status":"UNKNOWN" if face is None else "FACING" if face>=.60 else "NOT_FACING","hand_near_body_score":hs,"hand_near_body":hs>0,"close_duration":s["close"],"approach_duration":s["approach"],"following_duration":s["follow"],"hand_proximity_duration":s["hand"],"interaction_score":min(1.,.24*prox+.18*approach+.18*following+.14*(face or 0)+.16*hs+.10*persistence),"detection_confidence":(float(a.get("confidence",0))+float(b.get("confidence",0)))/2})
        self.states={k:v for k,v in self.states.items() if ts-v["last"]<=3.}; return {"timestamp":ts,"tracked_persons":len(people),"pairs":output}


class M5Application:
    """Three independently movable Toplevel windows backed by a latest-result pipeline."""
    def __init__(self, config: Optional[M5Config] = None, feature_source: Optional[Callable[[], Mapping[str, Any]]] = None) -> None:
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.config = tk, ttk, config or M5Config()
        self.root = tk.Tk(); self.root.withdraw(); self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.engine, self.feature_source = RulesEngineM5(self.config), feature_source
        self.m6_config=M6Config(); self.session_id=""; self.camera_id=""; self.active_source_type=""; self.active_source=""; self.recorder: Optional[RollingEvidenceRecorder]=None
        self.database=DatabaseWorker(self.m6_config); self.events=EventManager(self.m6_config,self.database)
        self.database.start(); self.events.start()
        self.detector, self.tracker, self.interactions = M4CompatibleDetector(), M4CompatibleTracker(), M4CompatibleInteractions()
        self.metadata_config=PersonMetadataConfig(); self.metadata=PersonMetadataManager(self.metadata_config); self.metadata.start()
        self.protected_targets=ProtectedTargetManager()
        self.stop_event, self.paused = threading.Event(), threading.Event()
        self.results, self.frames, self.processed_frames, self.diagnostics = LatestSlot(), LatestSlot(), LatestSlot(), LatestSlot()
        self.protected_interactions = LatestSlot()
        self.worker: Optional[threading.Thread] = None; self.capture_worker: Optional[threading.Thread] = None
        self.capture: Any = None; self.capture_lock = threading.Lock(); self.closed = False
        self.module_enabled = {"Person Detection": True, "Multi-Person Tracking": True, "Interaction Analysis": True, "Rules Engine": True, "Person Metadata": True}
        self.source_fps, self.frames_received, self.capture_started = 0.0, 0, 0.0
        self.tuning_defaults = self.engine.configuration(); self.tuning_log: Deque[str] = deque(maxlen=80)
        self._build_control(); self._build_preview(); self._build_data(); self._build_tuning(); self._build_metadata_window(); self._build_protected_target_window()
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
        self.status = tk.StringVar(value="Ready — protected-target behavior analysis available")
        ttk.Label(frame, text="AI CAMERA CONTROL", font=("Arial", 13, "bold")).pack(anchor="w")
        row = ttk.Frame(frame); row.pack(fill="x", pady=5)
        ttk.Label(row, text="Input:").pack(side="left"); ttk.Combobox(row, textvariable=self.source_kind, values=("Local Webcam", "IP Webcam", "Video File"), state="readonly", width=14).pack(side="left", padx=4)
        ttk.Entry(row, textvariable=self.source_value, width=36).pack(side="left", padx=4)
        ttk.Button(row, text="Browse", command=self._choose_file).pack(side="left")
        evidence=ttk.Frame(frame); evidence.pack(fill="x",pady=4); self.evidence_status=tk.StringVar(value="Evidence: ready")
        ttk.Button(evidence,text="Open Evidence Folder",command=self._open_evidence_folder).pack(side="left"); ttk.Label(evidence,textvariable=self.evidence_status).pack(side="left",padx=8)
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
        self.event_tree = ttk.Treeview(frame, columns=("Time", "Pair", "Score", "State"), show="headings", height=6)
        for col, width in (("Time", 110), ("Pair", 110), ("Score", 80), ("State", 120)):
            self.event_tree.heading(col, text=col); self.event_tree.column(col, width=width, anchor="center")
        self.event_tree.pack(fill="x")

    def _build_metadata_window(self) -> None:
        """The app uses separate windows rather than Notebook tabs; keep that pattern."""
        tk,ttk=self.tk,self.ttk; self.metadata_window=self._window("PERSON METADATA","620x500")
        outer=ttk.Frame(self.metadata_window,padding=9); outer.pack(fill="both",expand=True)
        ttk.Label(outer,text="PERSON METADATA — MANUAL ONLY",font=("Arial",11,"bold")).pack(anchor="w")
        ttk.Label(outer,text="No sensitive attributes are inferred from video. Enter only an authorized neutral note.",wraplength=580).pack(anchor="w",pady=(4,6))
        settings=ttk.LabelFrame(outer,text="Authorized operator entry",padding=5); settings.pack(fill="x",pady=6)
        self.metadata_track_var=tk.StringVar(); self.metadata_note_var=tk.StringVar()
        ttk.Label(settings,text="Track ID").grid(row=0,column=0,sticky="w"); ttk.Entry(settings,textvariable=self.metadata_track_var,width=10).grid(row=0,column=1,sticky="w",padx=4)
        ttk.Label(settings,text="Neutral label/note").grid(row=0,column=2,sticky="w",padx=(10,4)); ttk.Entry(settings,textvariable=self.metadata_note_var,width=32).grid(row=0,column=3,sticky="w")
        ttk.Button(settings,text="Apply Manual Metadata",command=self._apply_manual_metadata).grid(row=1,column=0,columnspan=4,sticky="w",pady=(6,0))
        self.metadata_status=tk.StringVar(value="Metadata source: Unavailable until entered manually")
        ttk.Label(outer,textvariable=self.metadata_status).pack(anchor="w")
        self.metadata_tree=ttk.Treeview(outer,columns=("Track ID","Metadata","Source","Status"),show="headings",height=12)
        for col,width in (("Track ID",80),("Metadata",220),("Source",110),("Status",160)): self.metadata_tree.heading(col,text=col); self.metadata_tree.column(col,width=width,anchor="center")
        self.metadata_tree.pack(fill="both",expand=True)

    def _apply_manual_metadata(self) -> None:
        if self.metadata.set_manual_metadata(self.metadata_track_var.get(), self.metadata_note_var.get()):
            self.metadata_note_var.set(""); self.metadata_status.set("Manual metadata saved for authorized operator review")
        else:
            self.metadata_status.set("Enter a valid Track ID and non-empty neutral note")

    def _build_protected_target_window(self) -> None:
        tk, ttk = self.tk, self.ttk
        self.protected_window = self._window("PROTECTED TARGET", "900x500")
        outer = ttk.Frame(self.protected_window, padding=9); outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="PROTECTED TARGET", font=("Arial", 11, "bold")).pack(anchor="w")
        controls = ttk.Frame(outer); controls.pack(fill="x", pady=6)
        self.protected_id_var = tk.StringVar(); self.show_protected_var = tk.BooleanVar(value=True)
        self.show_protected_overlay = True
        ttk.Label(controls, text="Track ID").pack(side="left")
        ttk.Entry(controls, textvariable=self.protected_id_var, width=10).pack(side="left", padx=4)
        ttk.Button(controls, text="Set Protected Target", command=self._set_protected_target).pack(side="left", padx=4)
        ttk.Button(controls, text="Clear Target", command=self._clear_protected_target).pack(side="left", padx=4)
        ttk.Checkbutton(controls, text="Show Protected Target", variable=self.show_protected_var, command=self._apply_protected_overlay).pack(side="left", padx=12)
        self.protected_status = tk.StringVar(value="No protected target selected")
        ttk.Label(outer, textvariable=self.protected_status).pack(anchor="w")
        self.protected_tree = ttk.Treeview(outer, columns=("Other ID", "Distance", "Proximity", "Approach", "Following", "Facing", "Target Moving Away", "Duration", "Rule State"), show="headings", height=14)
        for col, width in (("Other ID",70),("Distance",85),("Proximity",95),("Approach",95),("Following",85),("Facing",85),("Target Moving Away",130),("Duration",85),("Rule State",100)):
            self.protected_tree.heading(col, text=col); self.protected_tree.column(col, width=width, anchor="center")
        self.protected_tree.pack(fill="both", expand=True)

    def _set_protected_target(self) -> None:
        if self.protected_targets.set_target(self.protected_id_var.get()):
            self.protected_status.set(f"Protected Target ID: {self.protected_id_var.get()}")
        else:
            self.protected_status.set("Enter a valid existing Track ID")

    def _clear_protected_target(self) -> None:
        self.protected_targets.clear_target(self.protected_id_var.get())
        self.protected_status.set("No protected target selected")

    def _apply_protected_overlay(self) -> None:
        self.show_protected_overlay = bool(self.show_protected_var.get())

    def _build_tuning(self) -> None:
        tk, ttk = self.tk, self.ttk; self.tuning_window=self._window("RULE PARAMETER TUNING","760x760")
        outer=ttk.Frame(self.tuning_window,padding=8); outer.pack(fill="both",expand=True)
        self.tuning_summary=tk.StringVar(value="LIVE TUNING PREVIEW — waiting for an interaction")
        ttk.Label(outer,textvariable=self.tuning_summary,font=("Arial",11,"bold")).pack(anchor="w")
        self.weight_total=tk.StringVar(); ttk.Label(outer,textvariable=self.weight_total).pack(anchor="w")
        canvas=tk.Canvas(outer,highlightthickness=0); scroll=ttk.Scrollbar(outer,orient="vertical",command=canvas.yview); canvas.configure(yscrollcommand=scroll.set); scroll.pack(side="right",fill="y"); canvas.pack(fill="both",expand=True)
        body=ttk.Frame(canvas); canvas.create_window((0,0),window=body,anchor="nw"); body.bind("<Configure>",lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        self.tuning_vars={}
        self._tuning_section(body,"RULE WEIGHTS",[("weights",k,0.,40.,.1) for k in self.config.weights])
        scalars=[("close_persistence_seconds",0.,30.,.1),("duration_full_seconds",1.,120.,1.),("following_persistence_seconds",0.,30.,.1),("rapid_approach_score",0.,1.,.01),("escalation_distance_delta",0.,3.,.01),("high_speed",0.,400.,1.),("ema_alpha",.01,1.,.01),("episode_gap_seconds",0.,60.,.5),("alert_cooldown_seconds",0.,300.,1.)]
        self._tuning_section(body,"BEHAVIOR / TEMPORAL",[("scalar",n,lo,hi,step) for n,lo,hi,step in scalars])
        self._tuning_section(body,"STATE ENTRY",[("enter_thresholds",n,0.,100.,1.) for n in self.config.enter_thresholds])
        self._tuning_section(body,"HYSTERESIS EXIT",[("exit_thresholds",n,0.,100.,1.) for n in self.config.exit_thresholds])
        ttk.Label(body,text="LIVE CONTRIBUTIONS",font=("Arial",10,"bold")).pack(anchor="w",pady=(8,2)); self.contributions=tk.Text(body,height=10,wrap="none",state="disabled"); self.contributions.pack(fill="x")
        ttk.Label(body,text="TUNING HISTORY",font=("Arial",10,"bold")).pack(anchor="w",pady=(8,2)); self.tuning_history=tk.Text(body,height=5,wrap="word",state="disabled"); self.tuning_history.pack(fill="x")
        buttons=ttk.Frame(body); buttons.pack(fill="x",pady=8)
        for text,cmd in (("RESET DEFAULTS",self._reset_tuning),("SAVE CONFIG",self._save_tuning),("LOAD CONFIG",self._load_tuning)):
            ttk.Button(buttons,text=text,command=cmd).pack(side="left",padx=3)

    def _tuning_section(self,parent: Any,title: str, rows: Sequence[Tuple[str,str,float,float,float]]) -> None:
        frame=self.ttk.LabelFrame(parent,text=title,padding=5); frame.pack(fill="x",pady=4)
        for section,name,lo,hi,step in rows:
            value=self.config.weights[name] if section=="weights" else getattr(self.config,section)[name] if section.endswith("thresholds") else getattr(self.config,name)
            var=self.tk.DoubleVar(value=value); label=self.tk.StringVar(value=f"{value:.2f}"); self.tuning_vars[(section,name)]=(var,label)
            row=self.ttk.Frame(frame); row.pack(fill="x",pady=1); self.ttk.Label(row,text=name.replace("_"," ").title(),width=29).pack(side="left")
            scale=self.tk.Scale(row,from_=lo,to=hi,resolution=step,orient="horizontal",variable=var,showvalue=False,length=280,command=lambda _v,s=section,n=name,v=var,l=label:self._tune_change(s,n,v,l)); scale.pack(side="left",fill="x",expand=True); self.ttk.Label(row,textvariable=label,width=7).pack(side="left")

    def _tune_change(self,section: str,name: str,var: Any,label: Any) -> None:
        value=round(float(var.get()),2)
        if self.engine.update_parameter(section,name,value):
            label.set(f"{value:.2f}"); self.tuning_log.appendleft(f"{time.strftime('%H:%M:%S')} {name}: {value:.2f}")
        else:
            current=self.config.weights[name] if section=="weights" else getattr(self.config,section)[name] if section.endswith("thresholds") else getattr(self.config,name); var.set(current); label.set(f"{current:.2f}"); self.tuning_log.appendleft(f"{time.strftime('%H:%M:%S')} rejected invalid {name}")

    def _reset_tuning(self) -> None:
        self.engine.load_configuration({"config":copy.deepcopy(self.tuning_defaults)}); self._sync_tuning_vars(); self.tuning_log.appendleft(f"{time.strftime('%H:%M:%S')} restored defaults")

    def _sync_tuning_vars(self) -> None:
        for (section,name),(var,label) in self.tuning_vars.items():
            value=self.config.weights[name] if section=="weights" else getattr(self.config,section)[name] if section.endswith("thresholds") else getattr(self.config,name); var.set(value); label.set(f"{value:.2f}")

    def _save_tuning(self) -> None:
        from tkinter import filedialog, messagebox
        path=filedialog.asksaveasfilename(defaultextension=".json",filetypes=[("JSON","*.json")],initialfile="rules_config.json")
        if path:
            import json
            with open(path,"w",encoding="utf-8") as f: json.dump({"version":1,"timestamp":time.time(),"config":self.engine.configuration()},f,indent=2)
            self.tuning_log.appendleft(f"{time.strftime('%H:%M:%S')} saved {Path(path).name}")

    def _load_tuning(self) -> None:
        from tkinter import filedialog, messagebox
        path=filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if path:
            import json
            try:
                with open(path,encoding="utf-8") as f: data=json.load(f)
                if not self.engine.load_configuration(data): raise ValueError("invalid values or threshold ordering")
                self._sync_tuning_vars(); self.tuning_log.appendleft(f"{time.strftime('%H:%M:%S')} loaded {Path(path).name}")
            except (OSError,ValueError,json.JSONDecodeError) as exc: messagebox.showerror("Load configuration",f"Configuration rejected: {exc}")

    def _choose_file(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All files", "*.*")])
        if path:
            self.source_kind.set("Video File"); self.source_value.set(path)

    def _open_evidence_folder(self) -> None:
        folder=Path(self.m6_config.evidence_directory); folder.mkdir(parents=True,exist_ok=True)
        try:
            os.startfile(str(folder))
        except AttributeError:
            LOGGER.info("Evidence folder: %s",folder.resolve())
        except OSError as exc:
            self.status.set(f"ERROR opening evidence folder: {exc}")

    def start(self) -> None:
        if self.capture_worker and self.capture_worker.is_alive():
            self.paused.clear(); self.status.set("RECEIVING FRAMES"); return
        kind=self.source_kind.get(); source = self.source_value.get().strip() if kind=="Local Webcam" else normalize_source(self.source_value.get(), kind)
        if kind=="Local Webcam":
            try: source=str(max(0,int(source or "0")))
            except ValueError: self.status.set("ERROR: local webcam index must be a non-negative integer."); return
        if not source or (kind == "Video File" and not Path(source).is_file()):
            self.status.set("ERROR: choose a readable video file or an IP Webcam URL."); return
        self.source_value.set(source); self.stop_event.clear(); self.paused.clear(); self.engine.reset(); self.capture_started = time.monotonic(); self.frames_received = 0
        self.session_id=f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"; self.camera_id=f"CAM_LOCAL_{source}" if kind=="Local Webcam" else "CAM_IP_01" if kind=="IP Webcam" else f"VIDEO_{Path(source).stem}"; self.active_source_type=kind; self.active_source=source
        self.recorder=RollingEvidenceRecorder(self.m6_config,self.session_id,self.events.add_segment) ; self.recorder.start()
        self.capture_worker = threading.Thread(target=self._capture_loop, args=(int(source) if kind=="Local Webcam" else source, kind), name="m6-camera-capture", daemon=True)
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
                if self.recorder: self.recorder.submit(frame,now)
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
                    self.protected_targets.update_tracks(people, frame_at)
                    # Independent metadata branch: never passed to M4/M5 scoring.
                    if self.module_enabled["Person Metadata"] and self.metadata_config.enabled:
                        self.metadata.submit_latest(frame, people, frame_at)
                    if self.module_enabled["Interaction Analysis"]:
                        upstream = self.interactions.process(people, frame_at)
                        protected_pairs = self.protected_targets.protected_pairs(upstream.get("pairs", ()))
                        for pair in protected_pairs:
                            target_id = pair["protected_target_id"]
                            pair["target_moving_away"] = pair.get("a_moving_away", False) if target_id == pair["person_a"] else pair.get("b_moving_away", False)
                            for source_pair in upstream.get("pairs", ()):
                                if source_pair.get("person_a") == pair["person_a"] and source_pair.get("person_b") == pair["person_b"]:
                                    source_pair.update({"protected_target_id": target_id, "other_person_id": pair["other_person_id"],
                                                        "target_moving_away": pair["target_moving_away"]})
                        self.protected_interactions.put({"timestamp": frame_at, "pairs": protected_pairs})
                    try:
                        import cv2
                        drawn = frame.copy()
                        for p in people:
                            x1,y1,x2,y2=p["bbox"]; track_id=p.get("person_id","?"); protected=self.show_protected_overlay and self.protected_targets.is_target(track_id); color=(0,165,255) if protected else (0,220,0); cv2.rectangle(drawn,(x1,y1),(x2,y2),color,2); cv2.putText(drawn,f"ID {track_id}",(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
                            if protected: cv2.putText(drawn,"PROTECTED TARGET",(x1,max(36,y1-22)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
                        self.processed_frames.put((drawn, frame_at))
                    except Exception as exc: LOGGER.warning("M5 overlay unavailable: %s",exc)
                if upstream and self.module_enabled["Rules Engine"]:
                    assessed=self.engine.evaluate_batch(upstream); self.results.put(assessed)
                    for risk in assessed["assessments"]: self.events.observe(risk,self.camera_id,self.active_source_type,self.active_source,self.session_id)
                    self.events.tick(time.monotonic())
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
        if self.recorder: self.recorder.stop(); self.recorder=None
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
            for iid in self.event_tree.get_children(): self.event_tree.delete(iid)
            for event in payload.get("recent_events", [])[:self.config.event_history_size]:
                self.event_tree.insert("", "end", values=(f"{event['timestamp']:.1f}", f"#{event['pair_ids'][0]} ↔ #{event['pair_ids'][1]}", f"{event['score']:.0f}", event["state"]))
            self._refresh_tuning(payload.get("highest_risk"))
        self.events.tick(time.monotonic())
        event_status=self.events.status.get(); db_status=self.database.status.get()
        active=self.events.current_event()
        if active: self.evidence_status.set(f"Evidence: {active.status} {active.event_id}")
        elif event_status: self.evidence_status.set(f"Evidence: {event_status['state']} {event_status['message']}")
        if db_status and db_status.get("state") == "DB UNAVAILABLE": self.status.set(f"{self.status.get()} | Database unavailable")
        self._refresh_metadata()
        self._refresh_protected_targets()
        try: self.root.after(self.config.gui_update_interval_ms, self._refresh)
        except self.tk.TclError: pass

    def _refresh_metadata(self) -> None:
        results=self.metadata.snapshot(); current=set()
        self.metadata_status.set("Metadata source: Manual operator entry only; video classification unavailable")
        for track_id,result in sorted(results.items()):
            iid=str(track_id); current.add(iid)
            values=(track_id,result["metadata"],result["source"],result["status"])
            if self.metadata_tree.exists(iid): self.metadata_tree.item(iid,values=values)
            else:self.metadata_tree.insert("","end",iid=iid,values=values)
        for iid in set(self.metadata_tree.get_children())-current:self.metadata_tree.delete(iid)

    def _refresh_protected_targets(self) -> None:
        pairs = (self.protected_interactions.get() or {}).get("pairs", ())
        for iid in self.protected_tree.get_children(): self.protected_tree.delete(iid)
        for pair in pairs:
            assessment = next((item for item in (self.results.get() or {}).get("assessments", ())
                               if tuple(item.get("pair_ids", ())) == (pair["person_a"], pair["person_b"])), None)
            self.protected_tree.insert("", "end", values=(pair["other_person_id"], f"{pair['normalized_distance']:.2f}", pair["proximity_level"],
                "Approaching" if pair.get("approaching") else "Separating" if pair.get("separating") else "Stable",
                "Yes" if pair.get("following") else "No", pair.get("facing_status", "Unknown"),
                "Yes" if pair.get("target_moving_away") else "No", f"{pair.get('close_duration', 0):.1f}s",
                assessment.get("risk_state", "Not evaluated") if assessment else "Not evaluated"))

    def _refresh_tuning(self, best: Optional[Mapping[str,Any]]) -> None:
        total=sum(self.config.weights.values()); self.weight_total.set(f"Weight total: {total:.1f} points (score remains capped at 100; weights are absolute point contributions)")
        if best:
            pair=best["pair_ids"]; self.tuning_summary.set(f"LIVE: Person #{pair[0]} ↔ Person #{pair[1]}   Score {best['suspiciousness_score']:.1f}/100   {best['risk_state']}")
            rows=["Rule                           Signal    Weight  Contribution"]
            ranked=sorted(best["rule_contributions"].items(),key=lambda x:x[1],reverse=True)
            for name,contribution in ranked: rows.append(f"{name.replace('_',' ').title():29} {best['normalized_signals'].get(name,0):.2f}      {best['rule_weights'][name]:5.1f}     {contribution:6.2f}")
            if best["combination_bonuses"]: rows.append("Bonuses: "+", ".join(f"{k} +{v:.1f}" for k,v in best["combination_bonuses"].items()))
            self.contributions.configure(state="normal"); self.contributions.delete("1.0","end"); self.contributions.insert("1.0","\n".join(rows)); self.contributions.configure(state="disabled")
        self.tuning_history.configure(state="normal"); self.tuning_history.delete("1.0","end"); self.tuning_history.insert("1.0","\n".join(self.tuning_log)); self.tuning_history.configure(state="disabled")

    def close(self) -> None:
        if self.closed: return
        self.closed = True; self.stop_event.set()
        with self.capture_lock:
            if self.capture is not None: self.capture.release()
        for worker in (self.capture_worker, self.worker):
            if worker and worker.is_alive() and worker is not threading.current_thread(): worker.join(timeout=2)
        if self.recorder: self.recorder.stop()
        self.metadata.stop()
        self.events.stop(); self.database.stop()
        for window in (getattr(self, "control", None), getattr(self, "preview_window", None), getattr(self, "data_window", None), getattr(self, "tuning_window", None), getattr(self, "metadata_window", None), getattr(self, "protected_window", None)):
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
    targets=ProtectedTargetManager(ProtectedTargetConfig(cache_timeout_seconds=2.)); targets.update_tracks([{"person_id":5},{"person_id":2}],10.)
    assert targets.set_target(5,10.) and not targets.set_target(99,10.)
    protected=targets.protected_pairs([{"person_a":2,"person_b":5,"a_moving_away":False,"b_moving_away":True}])
    assert protected and protected[0]["protected_target_id"]==5 and protected[0]["target_moving_away"]
    targets.update_tracks([],11.); assert targets.is_target(5); targets.update_tracks([],13.1); assert not targets.is_target(5)
    # M6 lifecycle is deliberately asynchronous: no video/DB dependency in this unit check.
    cfg=M6Config(pre_event_seconds=300.,post_event_seconds=2.); db=DatabaseWorker(cfg); events=EventManager(cfg,db)
    events.add_segment(SegmentInfo(Path("synthetic.mp4"),1.,10.,1,20.,(1,1),"TEST"))
    risk={"risk_state":"SUSPICIOUS","timestamp":12.,"episode_id":"test-episode","pair_ids":(17,22),"suspiciousness_score":55.,"triggered_rule_ids":["HIGH_FOLLOWING"],"normalized_signals":{},"rule_contributions":{},"rule_weights":{}}
    events.observe(risk,"CAM_TEST","Video File","synthetic.mp4","TEST"); assert events.current_event() and events.current_event().pre_segments
    events.tick(15.); assert events.current_event() is None and not events.finalize.empty()
    print("M5/M6 synthetic tests passed")


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
