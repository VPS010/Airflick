import platform
import threading
import time

try:
    import pyautogui  # Lightweight and cross-platform for keyboard events
except ImportError:
    pyautogui = None

class ScreenshotTrigger:
    """
    Triggers a screenshot (Ctrl+PrtSc) when all five fingers are detected as touching/close ("pinch all" gesture).
    Works cross-platform (Windows, Linux). Does NOT save the image, just triggers the OS screenshot tool.
    Lightweight and non-blocking (runs in a thread).
    """
    def __init__(self, hand_detector):
        self.hand_detector = hand_detector
        self.triggered = False
        self.last_trigger_time = 0
        self.cooldown = 2  # seconds between triggers to avoid spamming

    def check_and_trigger(self, landmarks):
        """Return True if a screenshot gesture was detected and the hotkey was dispatched.

        Args:
            landmarks: List of hand landmark objects from MediaPipe.

        The method also rate-limits firing the hotkey using ``self.cooldown`` to
        prevent accidental screenshot spamming.
        """
        detected = False
        if self.is_all_fingers_pinch(landmarks):
            now = time.time()
            if not self.triggered or (now - self.last_trigger_time > self.cooldown):
                self.triggered = True
                self.last_trigger_time = now
                detected = True
                threading.Thread(target=self.send_screenshot_hotkey, daemon=True).start()
        else:
            self.triggered = False
        return detected

    def calculate_distance(self, point1, point2):
        """Calculate normalized Euclidean distance between two landmarks."""
        return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2) ** 0.5

    def is_all_fingers_pinch(self, landmarks):
        """
        Returns True if all 5 fingertips are close together ("pinch all" gesture).
        Uses pairwise Euclidean distance with a threshold of 0.1. Prints debug info.
        """
        if not landmarks or len(landmarks) < 21:
            return False
        tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]  # Thumb, index, middle, ring, pinky tips
        distances = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                d = self.calculate_distance(tips[i], tips[j])
                distances.append(d)
                if d >= 0.1:
                    print(f"[DEBUG] Fingertip pair {i}-{j} dist: {d:.4f} (too large)")
                    return False
        print(f"[DEBUG] All fingertip distances: {[f'{d:.4f}' for d in distances]}")
        print(f"[DEBUG] Min dist: {min(distances):.4f}, Max dist: {max(distances):.4f}")
        print("[DEBUG] Screenshot gesture detected!")
        return True

    def send_screenshot_hotkey(self):
        if pyautogui is None:
            print("pyautogui not installed, cannot trigger screenshot hotkey.")
            return
        os_name = platform.system()
        # Use PrintScreen for screenshot on Windows/Linux
        if os_name in ("Windows", "Linux"):
            print("[DEBUG] Sending hotkey: printscreen")
            pyautogui.hotkey('printscreen')
        else:
            pyautogui.hotkey('command', 'shift', '4')  # Mac example
