# logic/attention_logic.py

from math import sqrt
from typing import Optional, Dict, Any


class AttentionLogic:
    def __init__(
        self,
        suspicious_frames_threshold: int = 20,
        direction_min_cosine: float = 0.35,
        horizontal_offset_threshold: int = 80,
        vertical_offset_threshold: int = 50,
        yaw_threshold: float = 10.0,
        pitch_threshold: float = 5.0,
        smoothing_alpha: float = 0.4,
    ):
        self.suspicious_frames_threshold = suspicious_frames_threshold
        self.direction_min_cosine = direction_min_cosine
        self.horizontal_offset_threshold = horizontal_offset_threshold
        self.vertical_offset_threshold = vertical_offset_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.smoothing_alpha = smoothing_alpha
        self.suspicious_counter = 0
        self.smoothed_attention_score = 0.0

    @staticmethod
    def _vector_match(origin, target, projected_end, min_cosine: float) -> Optional[float]:
        gaze_dx = projected_end[0] - origin[0]
        gaze_dy = projected_end[1] - origin[1]
        target_dx = target[0] - origin[0]
        target_dy = target[1] - origin[1]

        gaze_norm = sqrt(gaze_dx * gaze_dx + gaze_dy * gaze_dy)
        target_norm = sqrt(target_dx * target_dx + target_dy * target_dy)
        if gaze_norm < 1e-6 or target_norm < 1e-6:
            return None

        cosine = (gaze_dx * target_dx + gaze_dy * target_dy) / (gaze_norm * target_norm)
        if cosine < min_cosine:
            return cosine

        return cosine

    def is_head_directed_to_phone(
        self,
        face_data: Optional[Dict[str, Any]],
        phone_data: Optional[Dict[str, Any]],
        pose_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if face_data is None:
            return {"match": False, "reason": "no_face"}

        if phone_data is None:
            return {"match": False, "reason": "no_phone"}

        if pose_data is None:
            return {"match": False, "reason": "no_pose"}

        face_center_x, face_center_y = face_data["center"]
        phone_center_x, phone_center_y = phone_data["center"]

        dx = phone_center_x - face_center_x
        dy = phone_center_y - face_center_y

        yaw = pose_data["yaw"]
        pitch = pose_data["pitch"]

        nose_tip = pose_data.get("nose_tip")
        nose_end = pose_data.get("nose_end")
        if nose_tip is not None and nose_end is not None:
            cosine = self._vector_match(
                origin=nose_tip,
                target=phone_data["center"],
                projected_end=nose_end,
                min_cosine=self.direction_min_cosine,
            )
            if cosine is not None:
                return {
                    "match": cosine >= self.direction_min_cosine,
                    "reason": "nose_vector",
                    "dx": dx,
                    "dy": dy,
                    "yaw": yaw,
                    "pitch": pitch,
                    "direction_score": cosine,
                }

        if dx > self.horizontal_offset_threshold:
            horizontal_match = yaw > self.yaw_threshold
            horizontal_reason = "phone_right"
        elif dx < -self.horizontal_offset_threshold:
            horizontal_match = yaw < -self.yaw_threshold
            horizontal_reason = "phone_left"
        else:
            horizontal_match = True
            horizontal_reason = "phone_center_x"

        if dy > self.vertical_offset_threshold:
            vertical_match = pitch > self.pitch_threshold
            vertical_reason = "phone_lower"
        elif dy < -self.vertical_offset_threshold:
            vertical_match = pitch < -self.pitch_threshold
            vertical_reason = "phone_upper"
        else:
            vertical_match = True
            vertical_reason = "phone_center_y"

        match = horizontal_match and vertical_match

        return {
            "match": match,
            "reason": f"{horizontal_reason}|{vertical_reason}",
            "dx": dx,
            "dy": dy,
            "yaw": yaw,
            "pitch": pitch,
            "direction_score": None,
        }

    def update(
        self,
        face_data: Optional[Dict[str, Any]],
        phone_data: Optional[Dict[str, Any]],
        pose_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        direction_result = self.is_head_directed_to_phone(
            face_data=face_data,
            phone_data=phone_data,
            pose_data=pose_data,
        )

        raw_score = 1.0 if direction_result["match"] else 0.0
        direction_score = direction_result.get("direction_score")
        if direction_score is not None:
            raw_score = max(0.0, min(1.0, (direction_score + 1.0) / 2.0))

        self.smoothed_attention_score = (
            (self.smoothing_alpha * raw_score)
            + ((1.0 - self.smoothing_alpha) * self.smoothed_attention_score)
        )

        looking_at_phone = direction_result["match"] or self.smoothed_attention_score >= 0.6

        if looking_at_phone:
            self.suspicious_counter += 1
        else:
            self.suspicious_counter = max(0, self.suspicious_counter - 1)

        alert = self.suspicious_counter >= self.suspicious_frames_threshold

        return {
            "looking_at_phone": looking_at_phone,
            "suspicious_counter": self.suspicious_counter,
            "alert": alert,
            "reason": direction_result.get("reason", ""),
            "dx": direction_result.get("dx", 0),
            "dy": direction_result.get("dy", 0),
            "yaw": direction_result.get("yaw", 0.0),
            "pitch": direction_result.get("pitch", 0.0),
            "direction_score": direction_result.get("direction_score"),
            "smoothed_attention_score": self.smoothed_attention_score,
        }
