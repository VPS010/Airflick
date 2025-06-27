import cv2
from pynput.mouse import Button, Controller
import pyautogui
import numpy as np
import time

class MouseController:
    def __init__(self):
        self.mouse = Controller()
        self.screen_width, self.screen_height = pyautogui.size()
        self.prev_x, self.prev_y = None, None  # Previous finger position
        self.smooth_factor = 0.6  # Changed default to 0.6
        self.last_click_time = 0
        self.click_cooldown = 0.5  # seconds between clicks to prevent spamming
        self.scaling_factor = 4.0  # Changed default to 4.0 to match slider
        self.calibration_corners = {
            "top_left": (0.2, 0.2),
            "top_right": (0.8, 0.2),
            "bottom_left": (0.2, 0.8),
            "bottom_right": (0.8, 0.8)
        }
        self.previous_gesture = None
        self.gesture_hold_frames = 0
        self.required_hold_frames = 5
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.2
        self.scroll_amount = 2
        self.hand_detector = None

    def is_index_finger_only(self, landmarks):
        index_raised = self.is_finger_raised(landmarks, 8, 5)
        thumb_down = not self.is_finger_raised(landmarks, 4, 3)
        middle_down = not self.is_finger_raised(landmarks, 12, 9)
        ring_down = not self.is_finger_raised(landmarks, 16, 13)
        pinky_down = not self.is_finger_raised(landmarks, 20, 17)
        return index_raised and thumb_down and middle_down and ring_down and pinky_down

    def is_left_click(self, landmarks):
        thumb_tip = landmarks[4]
        middle_tip = landmarks[12]
        distance = self.calculate_distance(thumb_tip, middle_tip)
        if distance < 0.05:
            return True
        return False

    def is_right_click(self, landmarks):
        thumb_tip = landmarks[4]
        ring_tip = landmarks[16]
        distance = self.calculate_distance(thumb_tip, ring_tip)
        if distance < 0.05:
            return True
        return False
    
    def calculate_distance(self, point1, point2):
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_raised(self, landmarks, tip_idx, pip_idx):
        return landmarks[tip_idx].y < landmarks[pip_idx].y
    
    def get_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
        
    def perform_click(self, button_type="left"):
        current_time = time.time()
        if current_time - self.last_click_time > self.click_cooldown:
            if button_type == "left":
                self.mouse.press(Button.left)
                self.mouse.release(Button.left)
            elif button_type == "right":
                self.mouse.press(Button.right)
                self.mouse.release(Button.right)
            self.last_click_time = current_time
            return True
        return False
    
    def perform_scroll(self, direction="up"):
        current_time = time.time()
        if current_time - self.last_scroll_time > self.scroll_cooldown:
            if direction == "up":
                self.mouse.scroll(0, self.scroll_amount)
            elif direction == "down":
                self.mouse.scroll(0, -self.scroll_amount)
            self.last_scroll_time = current_time
            return True
        return False

    def move_mouse_relative(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return None, None
        delta_x = (x - self.prev_x)
        delta_y = (y - self.prev_y)
        movement_speed = np.sqrt(delta_x**2 + delta_y**2)
        slow_threshold = 0.003
        if movement_speed < slow_threshold:
            accel_factor = 1.0
        else:
            accel_factor = 1.0 + (movement_speed - slow_threshold) * 40.0
            accel_factor = min(accel_factor, 4.0)
        delta_x = delta_x * self.scaling_factor * self.screen_width * accel_factor
        delta_y = delta_y * self.scaling_factor * self.screen_height * accel_factor
        delta_x = delta_x * self.smooth_factor
        delta_y = delta_y * self.smooth_factor
        current_x, current_y = self.mouse.position
        new_x = int(current_x + delta_x)
        new_y = int(current_y + delta_y)
        new_x = max(0, min(new_x, self.screen_width))
        new_y = max(0, min(new_y, self.screen_height))
        self.prev_x, self.prev_y = x, y
        try:
            self.mouse.position = (new_x, new_y)
            return new_x, new_y
        except:
            return None, None
    
    def move_mouse(self, x, y):
        try:
            pyautogui.moveTo(x, y)
            return True
        except pyautogui.FailSafeException:
            return False

    def map_to_screen(self, x, y):
        if not all(self.calibration_corners.values()):
            return 0, 0
        top_left = self.calibration_corners["top_left"]
        top_right = self.calibration_corners["top_right"]
        bottom_left = self.calibration_corners["bottom_left"]
        bottom_right = self.calibration_corners["bottom_right"]
        center_x = (top_left[0] + top_right[0] + bottom_left[0] + bottom_right[0]) / 4
        center_y = (top_left[1] + top_right[1] + bottom_left[1] + bottom_right[1]) / 4
        dx = (x - center_x) * self.scaling_factor
        dy = (y - center_y) * self.scaling_factor
        scaled_x = center_x + dx
        scaled_y = center_y + dy
        screen_x = np.interp(scaled_x, [top_left[0], top_right[0]], [0, self.screen_width])
        screen_y = np.interp(scaled_y, [top_left[1], bottom_left[1]], [0, self.screen_height])
        return int(screen_x), int(screen_y)

    def detect_gestures(self, frame, hand_landmarks):
        current_gesture = None
        if hand_landmarks:
            if self.hand_detector.is_thumbs_up(hand_landmarks.landmark):
                current_gesture = "Scroll Up"
                cv2.putText(frame, "Scroll Up Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif self.hand_detector.is_thumbs_down(hand_landmarks.landmark):
                current_gesture = "Scroll Down"
                cv2.putText(frame, "Scroll Down Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif self.is_left_click(hand_landmarks.landmark):
                current_gesture = "Left Click"
                cv2.putText(frame, "Left Click Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif self.is_right_click(hand_landmarks.landmark):
                current_gesture = "Right Click"
                cv2.putText(frame, "Right Click Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if current_gesture == self.previous_gesture and current_gesture is not None:
                self.gesture_hold_frames += 1
                if self.gesture_hold_frames >= self.required_hold_frames:
                    if current_gesture == "Left Click":
                        if self.perform_click("left"):
                            cv2.putText(frame, "Left Click Performed!", (50, 140), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            self.gesture_hold_frames = 0
                            return frame, "Left Click"
                    elif current_gesture == "Right Click":
                        if self.perform_click("right"):
                            cv2.putText(frame, "Right Click Performed!", (50, 140), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            self.gesture_hold_frames = 0
                            return frame, "Right Click"
                    elif current_gesture == "Scroll Up":
                        if self.perform_scroll("up"):
                            cv2.putText(frame, "Scrolling Up!", (50, 140),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            return frame, "Scroll Up"
                    elif current_gesture == "Scroll Down":
                        if self.perform_scroll("down"):
                            cv2.putText(frame, "Scrolling Down!", (50, 140),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            return frame, "Scroll Down"
            else:
                self.gesture_hold_frames = 0
            self.previous_gesture = current_gesture
        return frame, None
        
    def reset_tracking(self):
        self.prev_x, self.prev_y = None, None
