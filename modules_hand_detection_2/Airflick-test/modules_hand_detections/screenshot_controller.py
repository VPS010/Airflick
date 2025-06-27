import pyautogui
import datetime
import numpy as np
import cv2

class ScreenshotController:
    def __init__(self):
        self.screenshot_hold_frames = 0
        self.required_screenshot_hold = 5  # Frames to hold gesture for stability

    def calculate_distance(self, point1, point2):
        """Calculate normalized Euclidean distance between two landmarks."""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def is_screenshot_gesture(self, landmarks):
        """Check if all finger tips are close together (open palm, fingers touching)."""
        # Get finger tip landmarks
        tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]  # Thumb, index, middle, ring, pinky tips
        
        # Check pairwise distances between all tips
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                if self.calculate_distance(tips[i], tips[j]) >= 0.1:
                    return False
        
        return True

    def take_screenshot(self):
        """Capture and save a screenshot with a timestamped filename."""
        screenshot = pyautogui.screenshot()
        now = datetime.datetime.now()
        filename = f"screenshot_{now.strftime('%Y%m%d_%H%M%S')}.png"
        screenshot.save(filename)
        print(f"Screenshot saved as {filename}")

    def detect_screenshot_gesture(self, frame, hand_landmarks):
        """Detect the screenshot gesture and annotate the frame."""
        gesture = None
        if hand_landmarks:
            if self.is_screenshot_gesture(hand_landmarks.landmark):
                self.screenshot_hold_frames += 1
                cv2.putText(frame, "Screenshot Detected", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Blue text
                if self.screenshot_hold_frames >= self.required_screenshot_hold:
                    self.take_screenshot()
                    gesture = "Screenshot"
                    self.screenshot_hold_frames = 0
            else:
                self.screenshot_hold_frames = 0
        else:
            self.screenshot_hold_frames = 0
        
        return frame, gesture