from math import sqrt
from typing import Optional, Dict, Any


class PhoneTracker:
    def __init__(self, max_missed_frames: int = 12):
        self.max_missed_frames = max_missed_frames
        self.tracked_phone: Optional[Dict[str, Any]] = None
        self.missed_frames = 0

    @staticmethod
    def _distance(point_a, point_b) -> float:
        dx = point_a[0] - point_b[0]
        dy = point_a[1] - point_b[1]
        return sqrt(dx * dx + dy * dy)

    @staticmethod
    def _score_candidate(
        candidate: Dict[str, Any],
        target_point,
        max_distance: float,
    ) -> float:
        distance = sqrt(
            (candidate["center"][0] - target_point[0]) ** 2
            + (candidate["center"][1] - target_point[1]) ** 2
        )
        normalized_distance = min(distance / max(max_distance, 1.0), 1.0)
        normalized_area = min(candidate.get("area", 0.0) / 40000.0, 1.0)
        confidence = candidate.get("confidence", 0.0)

        return (confidence * 0.55) + (normalized_area * 0.20) + ((1.0 - normalized_distance) * 0.25)

    def update(self, phone_candidates: list[Dict[str, Any]], face_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not phone_candidates:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed_frames:
                self.tracked_phone = None
                return None

            if self.tracked_phone is None:
                return None

            stale_phone = dict(self.tracked_phone)
            stale_phone["tracked"] = True
            stale_phone["stale"] = True
            stale_phone["missed_frames"] = self.missed_frames
            return stale_phone

        target_point = None
        if self.tracked_phone is not None:
            target_point = self.tracked_phone["center"]
        elif face_data is not None:
            target_point = face_data["center"]

        if target_point is None:
            selected = max(phone_candidates, key=lambda item: (item["confidence"], item["area"]))
        else:
            max_distance = max(
                [self._distance(item["center"], target_point) for item in phone_candidates] or [1.0]
            )
            selected = max(
                phone_candidates,
                key=lambda item: (
                    self._score_candidate(item, target_point, max_distance),
                    item["confidence"],
                    item["area"],
                ),
            )

        selected = dict(selected)
        selected["tracked"] = True
        selected["stale"] = False
        selected["missed_frames"] = 0
        self.tracked_phone = selected
        self.missed_frames = 0
        return self.tracked_phone
