"""External gender-context adapter for tracked people.

This module does not infer gender from images. It accepts labels supplied by an
approved, consented external source and keeps them associated with Track IDs.
Missing, stale, or low-confidence values are always represented as Unknown.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


VALID_LABELS = frozenset({"Male", "Female", "Unknown"})


@dataclass(frozen=True)
class GenderEstimate:
    track_id: int
    gender: str = "Unknown"
    confidence: float = 0.0
    status: str = "Unknown"
    timestamp: float = 0.0


class ExternalGenderContext:
    """Track-ID cache for externally supplied, consented metadata."""

    def __init__(self, confidence_threshold: float = 0.70, timeout_seconds: float = 3.0) -> None:
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.timeout_seconds = max(0.0, timeout_seconds)
        self._estimates: Dict[int, GenderEstimate] = {}

    def update(self, values: Mapping[int, Mapping[str, Any]], timestamp: Optional[float] = None) -> None:
        now = time.time() if timestamp is None else timestamp
        for raw_id, value in values.items():
            try:
                track_id = int(raw_id)
                confidence = float(value.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            label = str(value.get("gender", "Unknown")).title()
            if label not in VALID_LABELS or confidence < self.confidence_threshold:
                label, status = "Unknown", "Low Confidence"
            else:
                status = str(value.get("status", "Stable"))
            self._estimates[track_id] = GenderEstimate(track_id, label, confidence, status, now)
        self.prune(now)

    def prune(self, timestamp: Optional[float] = None) -> None:
        now = time.time() if timestamp is None else timestamp
        self._estimates = {
            track_id: estimate
            for track_id, estimate in self._estimates.items()
            if now - estimate.timestamp <= self.timeout_seconds
        }

    def for_track(self, track_id: Any, timestamp: Optional[float] = None) -> GenderEstimate:
        self.prune(timestamp)
        try:
            track_id = int(track_id)
        except (TypeError, ValueError):
            return GenderEstimate(-1)
        return self._estimates.get(track_id, GenderEstimate(track_id))

    def annotate_people(self, people: Sequence[Mapping[str, Any]], timestamp: Optional[float] = None) -> list[Dict[str, Any]]:
        return [{**person, "gender": self.for_track(person.get("person_id"), timestamp).gender,
                 "gender_confidence": self.for_track(person.get("person_id"), timestamp).confidence}
                for person in people]

    def annotate_pairs(self, pairs: Sequence[Mapping[str, Any]], timestamp: Optional[float] = None) -> list[Dict[str, Any]]:
        output = []
        for pair in pairs:
            a = self.for_track(pair.get("person_a"), timestamp)
            b = self.for_track(pair.get("person_b"), timestamp)
            pair_type = f"{a.gender.lower()}_{b.gender.lower()}"
            if {a.gender, b.gender} == {"Male", "Female"}:
                pair_type = "male_female"
            output.append({**pair, "person_a_gender": a.gender, "person_b_gender": b.gender,
                           "person_a_gender_confidence": a.confidence,
                           "person_b_gender_confidence": b.confidence,
                           "gender_context": pair_type})
        return output

    def snapshot(self, timestamp: Optional[float] = None) -> Dict[int, Dict[str, Any]]:
        self.prune(timestamp)
        return {track_id: estimate.__dict__.copy() for track_id, estimate in self._estimates.items()}
