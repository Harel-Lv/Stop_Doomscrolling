# detection/face_detector.py

from typing import Optional, Dict, Any

import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_face(self, frame) -> Optional[Dict[str, Any]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        points = []
        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        bbox = (min(xs), min(ys), max(xs), max(ys))
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

        return {
            "landmarks": points,
            "bbox": bbox,
            "center": center,
        }

    def close(self) -> None:
        self.face_mesh.close()