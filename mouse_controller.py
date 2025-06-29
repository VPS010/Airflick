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
        self.click_cooldown = 0.1  # Further reduced for maximum responsiveness

        # Add scaling factor to amplify hand movements
        self.scaling_factor = 4.0  # Default sensitivity increased from 2.0 to 4.0

        # For gesture state tracking
        self.previous_gesture = None
        self.gesture_hold_frames = 0
        self.required_hold_frames = 0  # Removed hold time for instant gestures
        
        # For scrolling
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.03  # Faster scroll interval
        self.scroll_amount = 1  # Base scroll amount
        self.scroll_speed_factor = 1.0  # Default scroll speed factor
        self.scroll_accumulator = 0.0
        self.last_scroll_direction = None
        
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
        Check if gesture is left click (index finger pinched to thumb).
        This uses the distance between the thumb tip and the index finger tip.
        """
        # Get landmarks for thumb tip and index finger tip
        thumb_tip = landmarks[4]  # THUMB_TIP
        index_tip = landmarks[8]  # INDEX_FINGER_TIP

        # Calculate the distance between them
        distance = self.calculate_distance(thumb_tip, index_tip)

        # If the distance is very small, it's a click
        # Using a small threshold for precision to avoid accidental clicks
        if distance < 0.04:
            return True
        return False

    def is_right_click(self, landmarks):
        """
        Check if gesture is right click (middle finger pinched to thumb).
        This uses the distance between the thumb tip and the middle finger tip.
        """
        # Get landmarks for thumb tip and middle finger tip
        thumb_tip = landmarks[4]   # THUMB_TIP
        middle_tip = landmarks[12] # MIDDLE_FINGER_TIP

        # Calculate the distance between them
        distance = self.calculate_distance(thumb_tip, middle_tip)

        # If the distance is very small, it's a click
        if distance < 0.04:
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
            if self.last_scroll_direction != direction:
                self.scroll_accumulator = 0.0
                self.last_scroll_direction = direction

            self.scroll_accumulator += self.scroll_amount * self.scroll_speed_factor
            scroll_steps = int(self.scroll_accumulator)

            if scroll_steps > 0:
                if direction == "up":
                    self.mouse.scroll(0, scroll_steps)
                elif direction == "down":
                    self.mouse.scroll(0, -scroll_steps)
                
                self.scroll_accumulator -= scroll_steps
            
            self.last_scroll_time = current_time
            return True
        return False

    def set_scroll_speed(self, speed_factor):
        """Set the scroll speed factor."""
        self.scroll_speed_factor = float(speed_factor)

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