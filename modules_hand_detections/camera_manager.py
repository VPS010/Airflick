import cv2
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QImage

class CameraManager(QObject):
    frame_updated = pyqtSignal(QImage)
    gesture_detected = pyqtSignal(str)
    tracking_status_updated = pyqtSignal(str) # For messages like "Hand not detected" or "Tracking: Index Finger"

    def __init__(self, hand_detector, mouse_controller, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)
        
        self.hand_detector = hand_detector
        self.mouse_controller = mouse_controller
        
        self.is_tracking_active = False # Internal state for camera manager

    def start_feed(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        if self.cap and not self.cap.isOpened():
            print("Error: Could not open video stream.")
            self.cap = None 
            self.tracking_status_updated.emit("Error: Camera not found")
            return False
        if not self.timer.isActive():
            self.timer.start(30) # Approx 33 FPS
        return True

    def stop_feed(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        # Emit a signal or provide a way to clear the video feed in UI if needed

    def set_tracking_status(self, is_tracking):
        self.is_tracking_active = is_tracking

    def _process_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        
        # Process frame with hand detector
        processed_frame, hand_landmarks_list = self.hand_detector.find_hands(frame.copy()) # Use a copy for drawing
        
        current_gesture_text = "Gesture: None"
        tracking_info_text = "Status: Idle"

        if self.is_tracking_active:
            tracking_info_text = "Status: Tracking Active"
            if hand_landmarks_list:
                # Assuming we use the first detected hand
                hand_landmarks = hand_landmarks_list[0] 
                
                index_tip = hand_landmarks.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Move mouse
                new_x, new_y = self.mouse_controller.move_mouse_relative(index_tip.x, index_tip.y)

                if new_x is not None:
                    self.tracking_status_updated.emit("Tracking: Index Finger")
                
                # Add visual indicator for active tracking
                index_pos_viz = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                cv2.circle(processed_frame, index_pos_viz, 15, (0, 255, 0), -1) 
                cv2.putText(processed_frame, "TRACKING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Visualize the finger joints (optional, can be made configurable)
                index_dip_lm = hand_landmarks.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_DIP]
                index_pip_lm = hand_landmarks.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_PIP]
                index_mcp_lm = hand_landmarks.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_MCP]
                dip_pos = (int(index_dip_lm.x * frame.shape[1]), int(index_dip_lm.y * frame.shape[0]))
                pip_pos = (int(index_pip_lm.x * frame.shape[1]), int(index_pip_lm.y * frame.shape[0]))
                mcp_pos = (int(index_mcp_lm.x * frame.shape[1]), int(index_mcp_lm.y * frame.shape[0]))
                cv2.line(processed_frame, index_pos_viz, dip_pos, (255, 255, 0), 2)
                cv2.line(processed_frame, dip_pos, pip_pos, (255, 255, 0), 2)
                cv2.line(processed_frame, pip_pos, mcp_pos, (255, 255, 0), 2)

                # Detect clicks
                _, gesture = self.mouse_controller.detect_gestures(frame, hand_landmarks) # Pass original frame for gesture detection if needed
                if gesture:
                    current_gesture_text = f"Gesture: {gesture}"
                    self.gesture_detected.emit(gesture)
            
            else: # Hand is not detected, but tracking is ON
                self.mouse_controller.reset_tracking() 
                self.tracking_status_updated.emit("Gesture: Hand not detected")
        else:
            tracking_info_text = "Status: Tracking Stopped"
            # Optionally, could add a visual cue that tracking is off, or just show raw camera feed
            # For now, we just show the processed_frame which might have hand landmarks if detector finds them
            # but no tracking circle or gesture detection will run.
            if hand_landmarks_list: # Hand detected, but not tracking
                 self.tracking_status_updated.emit("Status: Hand Detected (Tracking Off)")
            else:
                 self.tracking_status_updated.emit("Status: No Hand Detected (Tracking Off)")


        # Convert frame to QImage for display
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.frame_updated.emit(qt_img)

    def __del__(self):
        self.stop_feed() # Ensure camera is released when object is deleted 