import cv2
from pynput.mouse import Button, Controller
import pyautogui
import numpy as np
import time
from hand_detection import HandDetector  # Import at the top level

class MouseController:
    def __init__(self):
        self.mouse = Controller()
        self.screen_width, self.screen_height = pyautogui.size()
        self.prev_x, self.prev_y = None, None  # Previous finger position
        self.smooth_factor = 0.2
        self.last_click_time = 0
        self.click_cooldown = 0.5  # seconds between clicks to prevent spamming
        
        # Add scaling factor to amplify hand movements
        self.scaling_factor = 2.0  # Adjust this value to change sensitivity
        
        # Calibration points (not needed for relative mode, but kept for compatibility)
        self.calibration_corners = {
            "top_left": (0.2, 0.2),
            "top_right": (0.8, 0.2),
            "bottom_left": (0.2, 0.8),
            "bottom_right": (0.8, 0.8)
        }
        
        # For gesture state tracking
        self.previous_gesture = None
        self.gesture_hold_frames = 0
        self.required_hold_frames = 5  # Number of frames to hold a gesture before triggering
        
        # For scrolling
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.2  # seconds between scroll actions
        self.scroll_amount = 2  # Number of "clicks" to scroll
        
        # Create a single instance of HandDetector to reuse
        self.hand_detector = HandDetector()

    def is_index_finger_only(self, landmarks):
        """Check if only index finger is up and all others are down"""
        # Check if index finger is raised
        index_raised = self.is_finger_raised(landmarks, 8, 5)  # Index finger
        
        # Check if other fingers are down
        thumb_down = not self.is_finger_raised(landmarks, 4, 3)  # Thumb
        middle_down = not self.is_finger_raised(landmarks, 12, 9)  # Middle finger
        ring_down = not self.is_finger_raised(landmarks, 16, 13)  # Ring finger
        pinky_down = not self.is_finger_raised(landmarks, 20, 17)  # Pinky finger
        
        return index_raised and thumb_down and middle_down and ring_down and pinky_down

    def is_left_click(self, landmarks):
        """
        Check if gesture is left click (middle finger pinched to thumb)
        This uses the distance between thumb tip and middle finger tip
        """
        # Get landmarks for thumb tip and middle tip
        thumb_tip = landmarks[4]
        middle_tip = landmarks[12]
        
        # Calculate distance between thumb tip and middle tip
        distance = self.calculate_distance(thumb_tip, middle_tip)
        
        # Check if other fingers are raised
        index_raised = self.is_finger_raised(landmarks, 8, 5)   # Index finger
        ring_raised = self.is_finger_raised(landmarks, 16, 13)  # Ring finger
        pinky_raised = self.is_finger_raised(landmarks, 20, 17) # Pinky finger
        
        # Left click gesture: thumb and middle are close
        if distance < 0.05:
            return True
        return False

    def is_right_click(self, landmarks):
        """
        Check if gesture is right click (thumb touches ring finger)
        """
        thumb_tip = landmarks[4]
        ring_tip = landmarks[16]
        
        distance = self.calculate_distance(thumb_tip, ring_tip)
        
        # Check finger positions for right click
        index_raised = self.is_finger_raised(landmarks, 8, 5)    # Index finger
        middle_raised = self.is_finger_raised(landmarks, 12, 9)  # Middle finger
        pinky_raised = self.is_finger_raised(landmarks, 20, 17)  # Pinky finger
        
        # Right click gesture: thumb and ring finger touch
        if distance < 0.05:
            return True
        return False
    
    def calculate_distance(self, point1, point2):
        """Calculate normalized distance between two landmarks"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_raised(self, landmarks, tip_idx, pip_idx):
        """Check if a finger is raised by comparing tip height with PIP joint"""
        return landmarks[tip_idx].y < landmarks[pip_idx].y
    
    def get_angle(self, a, b, c):
        """Calculate angle between three points (landmarks)"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # Ensure the value is in valid range for arccos
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
        
    def perform_click(self, button_type="left"):
        """Perform a mouse click"""
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
        """Perform a scroll action"""
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
        """
        Move mouse based on relative movement of finger
        Implements a mouse-like relative movement approach with acceleration
        """
        # Initialize previous position if this is the first call
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return None, None
        
        # Calculate movement delta
        delta_x = (x - self.prev_x) 
        delta_y = (y - self.prev_y)
        
        # Calculate movement speed (magnitude of the movement vector)
        movement_speed = np.sqrt(delta_x**2 + delta_y**2)
        
        # Apply pointer acceleration
        # Define thresholds for slow and fast movements
        slow_threshold = 0.003  # Adjust based on testing
        
        # Calculate acceleration factor based on movement speed
        if movement_speed < slow_threshold:
            # For slow movements: use base scaling for precision
            accel_factor = 1.0
        else:
            # For faster movements: apply non-linear acceleration
            # This creates a curve where faster movements get progressively more acceleration
            accel_factor = 1.0 + (movement_speed - slow_threshold) * 40.0  # Adjust multiplier based on testing
            # Cap the acceleration to avoid extreme jumps
            accel_factor = min(accel_factor, 4.0)  # Maximum 4x acceleration
        
        # Apply base scaling factor first
        delta_x = delta_x * self.scaling_factor * self.screen_width
        delta_y = delta_y * self.scaling_factor * self.screen_height
        
        # Then apply the acceleration factor
        delta_x = delta_x * accel_factor
        delta_y = delta_y * accel_factor
        
        # Apply smoothing (smoothing is applied after acceleration to maintain responsiveness)
        delta_x = delta_x * self.smooth_factor
        delta_y = delta_y * self.smooth_factor
        
        # Get current mouse position
        current_x, current_y = self.mouse.position
        
        # Calculate new position
        new_x = int(current_x + delta_x)
        new_y = int(current_y + delta_y)
        
        # Ensure we stay within screen boundaries
        new_x = max(0, min(new_x, self.screen_width))
        new_y = max(0, min(new_y, self.screen_height))
        
        # Update previous position
        self.prev_x, self.prev_y = x, y
        
        # Move the mouse
        try:
            self.mouse.position = (new_x, new_y)
            return new_x, new_y
        except:
            return None, None
    
    def move_mouse(self, x, y):
        """Legacy method for absolute positioning - kept for compatibility"""
        try:
            pyautogui.moveTo(x, y)
            return True
        except pyautogui.FailSafeException:
            return False

    def map_to_screen(self, x, y):
        """Legacy method for absolute mapping - kept for compatibility"""
        if not all(self.calibration_corners.values()):
            return 0, 0

        top_left = self.calibration_corners["top_left"]
        top_right = self.calibration_corners["top_right"]
        bottom_left = self.calibration_corners["bottom_left"]
        bottom_right = self.calibration_corners["bottom_right"]
        
        # Calculate the center of the calibration area
        center_x = (top_left[0] + top_right[0] + bottom_left[0] + bottom_right[0]) / 4
        center_y = (top_left[1] + top_right[1] + bottom_left[1] + bottom_right[1]) / 4
        
        # Calculate the distance from center and apply scaling factor
        dx = (x - center_x) * self.scaling_factor
        dy = (y - center_y) * self.scaling_factor
        
        # Apply the scaled displacement to the center
        scaled_x = center_x + dx
        scaled_y = center_y + dy
        
        # Map to screen coordinates
        screen_x = np.interp(scaled_x, [top_left[0], top_right[0]], [0, self.screen_width])
        screen_y = np.interp(scaled_y, [top_left[1], bottom_left[1]], [0, self.screen_height])
        
        return int(screen_x), int(screen_y)

    def detect_gestures(self, frame, hand_landmarks):
        """Detect and perform mouse clicks based on hand gestures with improved stability"""
        current_gesture = None
        
        if hand_landmarks:
            # Use the single instance of HandDetector created in __init__
            # instead of creating a new one for every frame
            
            # Check for thumbs up gesture (scroll up)
            if self.hand_detector.is_thumbs_up(hand_landmarks.landmark):
                current_gesture = "Scroll Up"
                cv2.putText(frame, "Scroll Up Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Check for thumbs down gesture (scroll down)
            elif self.hand_detector.is_thumbs_down(hand_landmarks.landmark):
                current_gesture = "Scroll Down"
                cv2.putText(frame, "Scroll Down Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Check for left click gesture
            elif self.is_left_click(hand_landmarks.landmark):
                current_gesture = "Left Click"
                cv2.putText(frame, "Left Click Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Check for right click gesture
            elif self.is_right_click(hand_landmarks.landmark):
                current_gesture = "Right Click"
                cv2.putText(frame, "Right Click Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Handle gesture state for stability
            if current_gesture == self.previous_gesture and current_gesture is not None:
                self.gesture_hold_frames += 1
                if self.gesture_hold_frames >= self.required_hold_frames:
                    # We've held the gesture long enough to trigger
                    if current_gesture == "Left Click":
                        if self.perform_click("left"):
                            cv2.putText(frame, "Left Click Performed!", (50, 140), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            # Reset counter after click to avoid multiple clicks
                            self.gesture_hold_frames = 0
                            return frame, "Left Click"
                    
                    elif current_gesture == "Right Click":
                        if self.perform_click("right"):
                            cv2.putText(frame, "Right Click Performed!", (50, 140), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            # Reset counter after click
                            self.gesture_hold_frames = 0
                            return frame, "Right Click"
                    
                    elif current_gesture == "Scroll Up":
                        if self.perform_scroll("up"):
                            cv2.putText(frame, "Scrolling Up!", (50, 140), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            # Don't reset the counter to allow continuous scrolling
                            return frame, "Scroll Up"
                    
                    elif current_gesture == "Scroll Down":
                        if self.perform_scroll("down"):
                            cv2.putText(frame, "Scrolling Down!", (50, 140), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            # Don't reset the counter to allow continuous scrolling
                            return frame, "Scroll Down"
            else:
                # Reset counter if gesture changed
                self.gesture_hold_frames = 0
                
            self.previous_gesture = current_gesture
        
        return frame, None
        
    def reset_tracking(self):
        """Reset tracking state when finger tracking starts/stops"""
        self.prev_x, self.prev_y = None, None