# logic/attention_logic.py

from typing import Optional, Dict, Any


class AttentionLogic:
    def __init__(self, suspicious_frames_threshold: int = 20):
        self.suspicious_frames_threshold = suspicious_frames_threshold
        self.suspicious_counter = 0

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

        # התאמה אופקית
        # אם הטלפון מימין - נצפה שהראש יפנה ימינה
        # אם הטלפון משמאל - נצפה שהראש יפנה שמאלה
        # אם הטלפון יחסית במרכז - לא נכפה התאמה חזקה
        if dx > 80:
            horizontal_match = yaw > 10
            horizontal_reason = "phone_right"
        elif dx < -80:
            horizontal_match = yaw < -10
            horizontal_reason = "phone_left"
        else:
            horizontal_match = True
            horizontal_reason = "phone_center_x"

        # התאמה אנכית
        # אם הטלפון נמוך משמעותית מהפנים - נצפה ל-pitch כלפי מטה
        if dy > 50:
            # לפעמים ב-solvePnP הסימן של pitch תלוי בקונבנציה,
            # לכן נרצה לבדוק בפועל ולכוון אם צריך.
            vertical_match = pitch < -5 or pitch > 5
            vertical_reason = "phone_lower"
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

        looking_at_phone = direction_result["match"]

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
        }