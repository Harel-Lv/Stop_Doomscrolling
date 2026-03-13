# utils/draw.py

import cv2


def draw_phone(frame, phone_data):
    if phone_data is None:
        return

    x1, y1, x2, y2 = phone_data["bbox"]
    conf = phone_data["confidence"]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        frame,
        f"Phone {conf:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cx, cy = phone_data["center"]
    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)


def draw_face(frame, face_data):
    if face_data is None:
        return

    x1, y1, x2, y2 = face_data["bbox"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cx, cy = face_data["center"]
    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    # מציירים רק חלק קטן מהנקודות כדי לא להעמיס
    for (x, y) in face_data["landmarks"][::25]:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def draw_pose(frame, pose_data):
    if pose_data is None:
        return

    text = (
        f"Yaw: {pose_data['yaw']:.1f}  "
        f"Pitch: {pose_data['pitch']:.1f}  "
        f"Roll: {pose_data['roll']:.1f}"
    )

    cv2.putText(
        frame,
        text,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    if "nose_tip" in pose_data and "nose_end" in pose_data:
        cv2.line(frame, pose_data["nose_tip"], pose_data["nose_end"], (0, 0, 255), 3)
        cv2.circle(frame, pose_data["nose_tip"], 4, (255, 0, 0), -1)


def draw_status(frame, logic_data, fps=None):
    if logic_data is None:
        return

    if logic_data["alert"]:
        status = "ALERT: LOOKING AT PHONE"
        color = (0, 0, 255)
    elif logic_data["looking_at_phone"]:
        status = "Suspicious"
        color = (0, 165, 255)
    else:
        status = "Focused"
        color = (0, 255, 0)

    cv2.putText(
        frame,
        status,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Counter: {logic_data['suspicious_counter']}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Reason: {logic_data.get('reason', '')}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        2
    )

    cv2.putText(
        frame,
        f"dx={logic_data.get('dx', 0):.0f}  dy={logic_data.get('dy', 0):.0f}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        2
    )

    cv2.putText(
        frame,
        f"yaw={logic_data.get('yaw', 0.0):.1f}  pitch={logic_data.get('pitch', 0.0):.1f}",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        2
    )

    if fps is not None:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )