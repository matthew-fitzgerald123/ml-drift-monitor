from __future__ import annotations
from river import drift
from dataclasses import dataclass, field


@dataclass
class FeatureDriftDetector:
    """
    Per-feature drift detector using ADWIN.
    ADWIN detects distribution shift in streaming data
    without requiring a fixed reference window.
    """
    feature_names: list[str]
    threshold: float = 0.05
    detectors: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in self.feature_names:
            self.detectors[name] = drift.ADWIN(delta=self.threshold)

    def update(self, feature_name: str, value: float) -> bool:
        """Feed one observation. Returns True if drift detected."""
        detector = self.detectors.get(feature_name)
        if detector is None:
            return False
        detector.update(value)
        return detector.drift_detected

    def update_all(self, features: dict[str, float]) -> dict[str, bool]:
        """Feed one full feature vector. Returns drift status per feature."""
        return {
            name: self.update(name, float(val))
            for name, val in features.items()
            if name in self.detectors
        }

    def reset(self, feature_name: str):
        if feature_name in self.detectors:
            self.detectors[feature_name] = drift.ADWIN(delta=self.threshold)

    def reset_all(self):
        for name in self.feature_names:
            self.reset(name)


class PredictionDriftDetector:
    """Monitors output distribution for shift over time."""

    def __init__(self):
        self.detector = drift.ADWIN(delta=0.05)

    def update(self, probability: float) -> bool:
        self.detector.update(probability)
        return self.detector.drift_detected

    def reset(self):
        self.detector = drift.ADWIN(delta=0.05)
