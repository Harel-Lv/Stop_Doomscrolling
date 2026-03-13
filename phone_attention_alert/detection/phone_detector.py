# detection/phone_detector.py

from typing import Optional, Dict, Any
from pathlib import Path
import os

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(Path(__file__).resolve().parents[1] / ".ultralytics")
)

from ultralytics import YOLO


class PhoneDetector:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.4,
        image_size: int = 960,
        max_candidates: int = 3,
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.image_size = image_size
        self.max_candidates = max_candidates

    def detect_phones(self, frame) -> list[Dict[str, Any]]:
        results = self.model(frame, verbose=False, imgsz=self.image_size)
        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        phone_candidates = []

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            class_name = self.model.names.get(cls_id, "")
            if class_name != "cell phone":
                continue

            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            area = max(0.0, (x2 - x1) * (y2 - y1))
            phone_candidates.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": (int(cx), int(cy)),
                "confidence": conf,
                "label": class_name,
                "area": area,
            })

        phone_candidates.sort(key=lambda item: (item["confidence"], item["area"]), reverse=True)
        return phone_candidates[:self.max_candidates]

    def detect_phone(self, frame) -> Optional[Dict[str, Any]]:
        candidates = self.detect_phones(frame)
        return candidates[0] if candidates else None
