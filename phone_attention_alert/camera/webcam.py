# camera/webcam.py

import cv2


class WebcamStream:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera at index {self.camera_index}")

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera is not opened")
        success, frame = self.cap.read()
        if not success:
            return None
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None