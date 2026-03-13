# pose/head_pose.py

from typing import Dict, Any, Optional
import numpy as np
import cv2


class HeadPoseEstimator:
    """
    מחשב head pose בעזרת solvePnP על בסיס MediaPipe Face Mesh landmarks.
    מחזיר yaw / pitch / roll בקירוב טוב יותר מהיוריסטיקה הפשוטה.
    """

    # MediaPipe landmark indices
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_LEFT_CORNER = 33
    RIGHT_EYE_RIGHT_CORNER = 263
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291

    def __init__(self, smoothing_alpha: float = 0.35):
        self.smoothing_alpha = smoothing_alpha
        self.previous_pose = None

    def _smooth_value(self, previous: float, current: float) -> float:
        alpha = self.smoothing_alpha
        return (alpha * current) + ((1.0 - alpha) * previous)

    def estimate(self, face_data: Optional[Dict[str, Any]], frame_shape=None) -> Optional[Dict[str, Any]]:
        if face_data is None or frame_shape is None:
            self.previous_pose = None
            return None

        landmarks = face_data["landmarks"]
        h, w = frame_shape[:2]

        try:
            image_points = np.array([
                landmarks[self.NOSE_TIP],
                landmarks[self.CHIN],
                landmarks[self.LEFT_EYE_LEFT_CORNER],
                landmarks[self.RIGHT_EYE_RIGHT_CORNER],
                landmarks[self.LEFT_MOUTH_CORNER],
                landmarks[self.RIGHT_MOUTH_CORNER],
            ], dtype=np.float64)
        except (IndexError, KeyError):
            self.previous_pose = None
            return None

        # מודל פנים תלת-ממדי סטנדרטי מקורב
        model_points = np.array([
            (0.0, 0.0, 0.0),          # nose tip
            (0.0, -63.6, -12.5),      # chin
            (-43.3, 32.7, -26.0),     # left eye left corner
            (43.3, 32.7, -26.0),      # right eye right corner
            (-28.9, -28.9, -24.1),    # left mouth corner
            (28.9, -28.9, -24.1),     # right mouth corner
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            self.previous_pose = None
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        nose_tip = tuple(map(int, image_points[0]))
        nose_3d_end, _ = cv2.projectPoints(
            np.array([(0.0, 0.0, 100.0)]),
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs
        )
        nose_end = tuple(map(int, nose_3d_end[0][0][:2]))

        pose_data = {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "nose_tip": nose_tip,
            "nose_end": nose_end,
            "rotation_vector": rotation_vector,
            "translation_vector": translation_vector,
        }

        if self.previous_pose is not None:
            pose_data["yaw"] = self._smooth_value(self.previous_pose["yaw"], pose_data["yaw"])
            pose_data["pitch"] = self._smooth_value(self.previous_pose["pitch"], pose_data["pitch"])
            pose_data["roll"] = self._smooth_value(self.previous_pose["roll"], pose_data["roll"])

            smoothed_end_x = self._smooth_value(self.previous_pose["nose_end"][0], pose_data["nose_end"][0])
            smoothed_end_y = self._smooth_value(self.previous_pose["nose_end"][1], pose_data["nose_end"][1])
            pose_data["nose_end"] = (int(smoothed_end_x), int(smoothed_end_y))

        self.previous_pose = {
            "yaw": pose_data["yaw"],
            "pitch": pose_data["pitch"],
            "roll": pose_data["roll"],
            "nose_end": pose_data["nose_end"],
        }

        return pose_data
