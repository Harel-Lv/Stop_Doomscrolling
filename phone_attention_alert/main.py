# main.py

import cv2
import time

from config import (
    CAMERA_INDEX,
    WINDOW_NAME,
    YOLO_MODEL_PATH,
    PHONE_CONFIDENCE_THRESHOLD,
    SUSPICIOUS_FRAMES_THRESHOLD,
)
from camera.webcam import WebcamStream
from detection.phone_detector import PhoneDetector
from detection.face_detector import FaceDetector
from pose.head_pose import HeadPoseEstimator
from logic.attention_logic import AttentionLogic
from alert.notifier import Notifier
from utils.draw import draw_phone, draw_face, draw_pose, draw_status


def main():
    webcam = WebcamStream(camera_index=CAMERA_INDEX)
    phone_detector = PhoneDetector(
        model_path=YOLO_MODEL_PATH,
        conf_threshold=PHONE_CONFIDENCE_THRESHOLD
    )
    face_detector = FaceDetector()
    pose_estimator = HeadPoseEstimator()
    attention_logic = AttentionLogic(
        suspicious_frames_threshold=SUSPICIOUS_FRAMES_THRESHOLD
    )
    notifier = Notifier(cooldown_seconds=2.0)

    webcam.open()

    prev_time = time.time()

    try:
        while True:
            frame = webcam.read()
            if frame is None:
                print("Failed to read frame from camera.")
                break

            # אפשר להקטין/להגדיל בעתיד אם צריך ביצועים
            frame = cv2.flip(frame, 1)

            # -------- Detection --------
            phone_data = phone_detector.detect_phone(frame)
            face_data = face_detector.detect_face(frame)
            pose_data = pose_estimator.estimate(face_data, frame.shape)

            # -------- Logic --------
            logic_data = attention_logic.update(
                face_data=face_data,
                phone_data=phone_data,
                pose_data=pose_data
            )

            # -------- Alert --------
            if logic_data["alert"]:
                notifier.trigger()

            # -------- FPS --------
            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            # -------- Draw --------
            draw_phone(frame, phone_data)
            draw_face(frame, face_data)
            draw_pose(frame, pose_data)
            draw_status(frame, logic_data, fps=fps)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        webcam.release()
        face_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()