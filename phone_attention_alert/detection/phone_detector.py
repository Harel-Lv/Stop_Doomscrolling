# detection/phone_detector.py

from typing import Optional, Dict, Any

from ultralytics import YOLO


class PhoneDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_phone(self, frame) -> Optional[Dict[str, Any]]:
        results = self.model(frame, verbose=False)
        if not results:
            return None

        result = results[0]
        if result.boxes is None:
            return None

        best_phone = None
        best_conf = 0.0

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

            if conf > best_conf:
                best_conf = conf
                best_phone = {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "center": (int(cx), int(cy)),
                    "confidence": conf,
                    "label": class_name,
                }

        return best_phone