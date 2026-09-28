"""Shared M3/M4/M5 pipeline adapter.

The adapter keeps one detection and tracking result flowing into interaction
features and risk assessments. Weapon inference is optional and requires a
custom YOLO model at ``models/weapon.pt``.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from interactive_analyser_M4 import DetectionManager, InteractionManager, TrackingManager
from gender_context import ExternalGenderContext
from rules_engine_M5 import M5Config, RulesEngineM5


class ProtectionPipeline:
    """Run detection, interaction analysis, and risk scoring as one pipeline."""

    def __init__(self, config: Optional[M5Config] = None) -> None:
        self.detector = DetectionManager()
        self.tracker = TrackingManager()
        self.interactions = InteractionManager()
        self.rules = RulesEngineM5(config)
        self.gender_context = ExternalGenderContext()

    def load_models(self) -> None:
        """Load person, pose, and optional weapon weights once."""
        self.detector.load_models()

    def process_people(
        self,
        people: Sequence[Mapping[str, Any]],
        timestamp: Optional[float] = None,
        weapons: Optional[Sequence[Mapping[str, Any]]] = None,
        gender_results: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Process already detected people, useful for replay and unit tests."""
        import time

        ts = time.time() if timestamp is None else timestamp
        if gender_results is not None:
            self.gender_context.update(gender_results, ts)
        tracked = self.tracker.update([dict(person) for person in people], ts)
        tracked = self.gender_context.annotate_people(tracked, ts)
        features = self.interactions.process(tracked, ts)
        features["pairs"] = self.gender_context.annotate_pairs(features.get("pairs", ()), ts)
        assessment = self.rules.evaluate_batch(features, timestamp=ts)
        return {
            "timestamp": ts,
            "persons": tracked,
            "weapons": list(weapons or []),
            "interactions": features,
            "risk": assessment,
        }

    def process_frame(self, frame: Any, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Run the complete pipeline for one BGR frame."""
        import time

        ts = time.time() if timestamp is None else timestamp
        people = self.detector.infer(frame, pose_enabled=True, detection_enabled=True)
        weapons = self.detector.infer_weapons(frame)
        return self.process_people(people, timestamp=ts, weapons=weapons)

    def close(self) -> None:
        """Stop background interaction work owned by this adapter."""
        self.interactions.stop_worker()
