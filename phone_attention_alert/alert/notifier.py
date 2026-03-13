# alert/notifier.py

import time
import platform


class Notifier:
    def __init__(self, cooldown_seconds: float = 2.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0.0

    def trigger(self) -> None:
        now = time.time()
        if now - self.last_alert_time < self.cooldown_seconds:
            return

        self.last_alert_time = now

        system_name = platform.system()

        if system_name == "Windows":
            try:
                import winsound
                winsound.Beep(1500, 300)
            except Exception:
                print("ALERT: Looking at phone!")
        else:
            print("ALERT: Looking at phone!")