import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QSlider
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6 import uic
import os

# Import our custom modules
from hand_detection import HandDetector
from mouse_controller import MouseController
from screenshot_controller import ScreenshotController

class AirFlick(QWidget):
    def __init__(self):
        super().__init__()
        
        # Load UI with correct path
        uic.loadUi(os.path.join(os.path.dirname(__file__), 'air_flick.ui'), self)
        
        # Initialize video capture
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Initialize our modules
        self.hand_detector = HandDetector()
        self.mouse_controller = MouseController()
        self.screenshot_controller = ScreenshotController()
        
        # Set default calibration values (no calibration needed)
        self.mouse_controller.calibration_corners = {
            "top_left": (0.2, 0.2),
            "top_right": (0.8, 0.2),
            "bottom_left": (0.2, 0.8),
            "bottom_right": (0.8, 0.8)
        }
        
        # Tracking states
        self.is_tracking = False
        self.is_calibrating = False
        self.current_calibration_step = 0
        self.calibration_steps = ["top_left", "top_right", "bottom_left", "bottom_right"]
        
        # Connect buttons
        self.startButton.clicked.connect(self.start_tracking)
        self.stopButton.clicked.connect(self.stop_all)
        # self.calibrateButton.clicked.connect(self.start_calibration)  # Commented out calibration
        self.settingsButton.clicked.connect(self.show_settings)
        
        # Setup sensitivity slider
        self.sensitivitySlider.setMinimum(10)
        self.sensitivitySlider.setMaximum(50)
        self.sensitivitySlider.setValue(int(self.mouse_controller.scaling_factor * 10))
        self.sensitivitySlider.valueChanged.connect(self.update_sensitivity)
        self.sensitivityValue.setText(f"{self.mouse_controller.scaling_factor:.1f}")
        
        # Setup smoothness slider
        self.smoothnessSlider.setMinimum(1)
        self.smoothnessSlider.setMaximum(10)
        self.smoothnessSlider.setValue(int(self.mouse_controller.smooth_factor * 10))
        self.smoothnessSlider.valueChanged.connect(self.update_smoothness)
        self.smoothnessValue.setText(f"{self.mouse_controller.smooth_factor:.1f}")
        
    def start_camera(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        if not self.timer.isActive():
            self.videoFeed.setStyleSheet("background-color: #1E1E1E;")
            self.timer.start(30)

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.videoFeed.clear()
        self.videoFeed.setStyleSheet("border: 2px solid #00D4FF; background-color: #1E1E1E;")

    def start_tracking(self):
        self.start_camera()
        self.is_tracking = True
        self.is_calibrating = False
        self.gestureOutput.setText("Gesture: Tracking started")

    def stop_all(self):
        self.stop_camera()
        self.is_tracking = False
        self.is_calibrating = False
        self.gestureOutput.setText("Gesture: None")

    def show_settings(self):
        self.gestureOutput.setText("Gesture: Settings button clicked")
        # Implement settings dialog here if needed

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                return

            frame = cv2.flip(frame, 1)
            
            # Process frame with hand detector
            frame, hand_landmarks = self.hand_detector.find_hands(frame)
            
            if hand_landmarks:
                hand_landmark = hand_landmarks[0]  # We're only using the first hand
                
                if self.is_tracking:
                    # Get index finger tip position
                    index_tip = hand_landmark.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_pos = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                    
                    # Get all index finger joints for angle visualization
                    index_dip = hand_landmark.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_DIP]
                    index_pip = hand_landmark.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_PIP]
                    index_mcp = hand_landmark.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    
                    dip_pos = (int(index_dip.x * frame.shape[1]), int(index_dip.y * frame.shape[0]))
                    pip_pos = (int(index_pip.x * frame.shape[1]), int(index_pip.y * frame.shape[0]))
                    mcp_pos = (int(index_mcp.x * frame.shape[1]), int(index_mcp.y * frame.shape[0]))
                    
                    # Calculate angles
                    angle1 = self.hand_detector.calculate_angle_between_points(index_tip, index_dip, index_pip)
                    angle2 = self.hand_detector.calculate_angle_between_points(index_dip, index_pip, index_mcp)
                    
                    # Check if index finger is straight
                    is_straight = self.hand_detector.is_index_finger_straight(hand_landmark.landmark)
                    
                    # Only move cursor when index finger is straight
                    if is_straight:
                        # Use relative movement approach - like a conventional mouse
                        new_x, new_y = self.mouse_controller.move_mouse_relative(index_tip.x, index_tip.y)
                        if new_x is not None:
                            self.gestureOutput.setText(f"Tracking: Delta X/Y relative to current position")
                        
                        # Add visual indicator for active tracking
                        cv2.circle(frame, index_pos, 15, (0, 255, 0), -1)  # Green circle for active tracking
                        cv2.putText(frame, "TRACKING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        # Not tracking because index finger is bent
                        self.gestureOutput.setText(f"Not tracking: Straighten finger more (angles: {angle1:.1f}°, {angle2:.1f}°)")
                        
                        # Reset relative tracking when finger is not straight
                        self.mouse_controller.reset_tracking()
                        
                        # Add visual indicator for inactive tracking
                        cv2.circle(frame, index_pos, 15, (0, 0, 255), 2)  # Red circle outline for inactive
                        cv2.putText(frame, "NOT TRACKING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Visualize the angles by drawing lines and showing angle values
                    cv2.line(frame, index_pos, dip_pos, (255, 255, 0), 2)
                    cv2.line(frame, dip_pos, pip_pos, (255, 255, 0), 2)
                    cv2.line(frame, pip_pos, mcp_pos, (255, 255, 0), 2)
                    
                    # Show angle values at each joint
                    cv2.putText(frame, f"{angle1:.1f}°", dip_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               (0, 255, 255) if angle1 > 175 else (0, 0, 255), 2)
                    cv2.putText(frame, f"{angle2:.1f}°", pip_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               (0, 255, 255) if angle2 > 175 else (0, 0, 255), 2)
                    
                    # Add indicator for required straightness
                    cv2.putText(frame, "Finger must be completely straight (>175°)",
                               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Detect mouse gestures
                    frame, mouse_gesture = self.mouse_controller.detect_gestures(frame, hand_landmark)
                    
                    # Detect screenshot gesture
                    frame, screenshot_gesture = self.screenshot_controller.detect_screenshot_gesture(frame, hand_landmark)
                    
                    # Update gestureOutput based on detected gestures
                    if screenshot_gesture:
                        self.gestureOutput.setText(f"Gesture: {screenshot_gesture}")
                    elif mouse_gesture:
                        self.gestureOutput.setText(f"Gesture: {mouse_gesture}")
                
            # Convert frame to QImage for display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_img = QImage(rgb_image.data, w, h, w * ch, QImage.Format.Format_RGB888)
            self.videoFeed.setPixmap(QPixmap.fromImage(qt_img))

    def update_sensitivity(self, value):
        # Convert slider value to scaling factor (10-50 → 1.0-5.0)
        scaling_factor = value / 10.0
        self.mouse_controller.scaling_factor = scaling_factor
        self.sensitivityValue.setText(f"{scaling_factor:.1f}")
    
    def update_smoothness(self, value):
        # Convert slider value to smoothness factor (1-10 → 0.1-1.0)
        smoothness = value / 10.0
        self.mouse_controller.smooth_factor = smoothness
        self.smoothnessValue.setText(f"{smoothness:.1f}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AirFlick()
    window.show()
    sys.exit(app.exec())