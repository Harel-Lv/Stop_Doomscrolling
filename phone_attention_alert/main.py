# main.py

import cv2
import time

if __package__ in (None, ""):
    from config import (
        CAMERA_INDEX,
        WINDOW_NAME,
        YOLO_MODEL_PATH,
        PHONE_CONFIDENCE_THRESHOLD,
        YOLO_IMAGE_SIZE,
        MAX_PHONE_CANDIDATES,
        SUSPICIOUS_FRAMES_THRESHOLD,
        PHONE_DIRECTION_MIN_COSINE,
        PHONE_HORIZONTAL_OFFSET_THRESHOLD,
        PHONE_VERTICAL_OFFSET_THRESHOLD,
        HEAD_POSE_YAW_THRESHOLD,
        HEAD_POSE_PITCH_THRESHOLD,
        POSE_SMOOTHING_ALPHA,
        ATTENTION_SMOOTHING_ALPHA,
        TRACKER_MAX_MISSED_FRAMES,
    )
    from camera.webcam import WebcamStream
    from detection.phone_detector import PhoneDetector
    from detection.face_detector import FaceDetector
    from pose.head_pose import HeadPoseEstimator
    from logic.attention_logic import AttentionLogic
    from alert.notifier import Notifier
    from utils.draw import draw_phone, draw_face, draw_pose, draw_status
else:
    from .config import (
        CAMERA_INDEX,
        WINDOW_NAME,
        YOLO_MODEL_PATH,
        PHONE_CONFIDENCE_THRESHOLD,
        YOLO_IMAGE_SIZE,
        MAX_PHONE_CANDIDATES,
        SUSPICIOUS_FRAMES_THRESHOLD,
        PHONE_DIRECTION_MIN_COSINE,
        PHONE_HORIZONTAL_OFFSET_THRESHOLD,
        PHONE_VERTICAL_OFFSET_THRESHOLD,
        HEAD_POSE_YAW_THRESHOLD,
        HEAD_POSE_PITCH_THRESHOLD,
        POSE_SMOOTHING_ALPHA,
        ATTENTION_SMOOTHING_ALPHA,
        TRACKER_MAX_MISSED_FRAMES,
    )
    from .camera.webcam import WebcamStream
    from .detection.phone_detector import PhoneDetector
    from .detection.face_detector import FaceDetector
    from .pose.head_pose import HeadPoseEstimator
    from .logic.attention_logic import AttentionLogic
    from .logic.phone_tracker import PhoneTracker
    from .alert.notifier import Notifier
    from .utils.draw import draw_phone, draw_face, draw_pose, draw_status
if __package__ in (None, ""):
    from logic.phone_tracker import PhoneTracker


def main():
    webcam = None
    face_detector = None

    try:
        webcam = WebcamStream(camera_index=CAMERA_INDEX)
        phone_detector = PhoneDetector(
            model_path=YOLO_MODEL_PATH,
            conf_threshold=PHONE_CONFIDENCE_THRESHOLD,
            image_size=YOLO_IMAGE_SIZE,
            max_candidates=MAX_PHONE_CANDIDATES,
        )
        face_detector = FaceDetector()
        pose_estimator = HeadPoseEstimator(smoothing_alpha=POSE_SMOOTHING_ALPHA)
        attention_logic = AttentionLogic(
            suspicious_frames_threshold=SUSPICIOUS_FRAMES_THRESHOLD,
            direction_min_cosine=PHONE_DIRECTION_MIN_COSINE,
            horizontal_offset_threshold=PHONE_HORIZONTAL_OFFSET_THRESHOLD,
            vertical_offset_threshold=PHONE_VERTICAL_OFFSET_THRESHOLD,
            yaw_threshold=HEAD_POSE_YAW_THRESHOLD,
            pitch_threshold=HEAD_POSE_PITCH_THRESHOLD,
            smoothing_alpha=ATTENTION_SMOOTHING_ALPHA,
        )
        phone_tracker = PhoneTracker(max_missed_frames=TRACKER_MAX_MISSED_FRAMES)
        notifier = Notifier(cooldown_seconds=2.0)

        webcam.open()
    except Exception as exc:
        print(f"Startup failed: {exc}")
        if face_detector is not None:
            face_detector.close()
        if webcam is not None:
            webcam.release()
        cv2.destroyAllWindows()
        return

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
            face_data = face_detector.detect_face(frame)
            phone_candidates = phone_detector.detect_phones(frame)
            tracked_phone = phone_tracker.update(phone_candidates, face_data=face_data)
            phone_data = tracked_phone if tracked_phone is not None and not tracked_phone.get("stale", False) else None
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
            draw_phone(frame, tracked_phone, phone_candidates=phone_candidates)
            draw_face(frame, face_data)
            draw_pose(frame, pose_data)
            draw_status(frame, logic_data, fps=fps)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        if webcam is not None:
            webcam.release()
        if face_detector is not None:
            face_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
