import sys
import cv2
import numpy as np
import os
import gc  # Import garbage collector
from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QMainWindow
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6 import uic

# Import our custom modules
from hand_detection import HandDetector
from mouse_controller import MouseController
from welcome_screen import WelcomeScreen

class AirFlick(QWidget):
    def __init__(self):
        super().__init__()
        
        # Load UI
        uic.loadUi('air_flick.ui', self)
        
        # Initialize video capture
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Initialize our modules
        self.hand_detector = HandDetector()
        self.mouse_controller = MouseController()
        
        # Memory management - use the hand_detector from MouseController to avoid duplicate instances
        self.mouse_controller.hand_detector = self.hand_detector
        
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
        
        # Setup memory management timer - force garbage collection every 60 seconds
        self.gc_timer = QTimer()
        self.gc_timer.timeout.connect(self.force_garbage_collection)
        self.gc_timer.start(60000)  # Run every 60 seconds
        
    def start_camera(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        if not self.timer.isActive():
            self.videoFeed.setStyleSheet("background-color: #0f172a;")
            self.timer.start(30)

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.videoFeed.clear()
        self.videoFeed.setStyleSheet("border: 2px solid #38bdf8; background-color: #0f172a;")

    def start_tracking(self):
        # No calibration check needed anymore
        # if not all(self.mouse_controller.calibration_corners.values()):
        #     self.gestureOutput.setText("Gesture: Please calibrate first")
        #     return
        self.start_camera()
        self.mouse_controller.reset_tracking() # Reset for a fresh start
        self.is_tracking = True
        self.is_calibrating = False
        self.gestureOutput.setText("Gesture: Tracking started")

    def stop_all(self):
        self.stop_camera()
        self.mouse_controller.reset_tracking() # Reset when stopping
        self.is_tracking = False
        self.is_calibrating = False
        self.gestureOutput.setText("Gesture: None")
        
        # Force garbage collection when stopping
        self.force_garbage_collection()

    # Memory management
    def force_garbage_collection(self):
        """Force Python garbage collection to free memory"""
        gc.collect()
        
    # Calibration methods (commented out but kept for reference)
    """
    def start_calibration(self):
        self.start_camera()
        self.is_calibrating = True
        self.is_tracking = False
        self.current_calibration_step = 0
        self.gestureOutput.setText(f"Calibration: Position hand at {self.calibration_steps[0]} and click")
    """
        
    def show_settings(self):
        self.gestureOutput.setText("Gesture: Settings button clicked")
        # Implement settings dialog here if needed

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                return

            # Create a copy to avoid OpenCV memory issues with frame references
            frame = frame.copy()
            frame = cv2.flip(frame, 1)
            
            # Process frame with hand detector
            processed_frame, hand_landmarks = self.hand_detector.find_hands(frame)
            
            # Let frame go out of scope for garbage collection
            frame = None
            
            if self.is_tracking: # Only process if tracking is globally enabled
                if hand_landmarks:
                    hand_landmark = hand_landmarks[0]  # We're only using the first hand
                    
                    # Get index finger tip position
                    index_tip = hand_landmark.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_pos = (int(index_tip.x * processed_frame.shape[1]), int(index_tip.y * processed_frame.shape[0]))
                    
                    # Get all index finger joints for drawing lines
                    index_dip_lm = hand_landmark.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_DIP]
                    index_pip_lm = hand_landmark.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_PIP]
                    index_mcp_lm = hand_landmark.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    
                    dip_pos = (int(index_dip_lm.x * processed_frame.shape[1]), int(index_dip_lm.y * processed_frame.shape[0]))
                    pip_pos = (int(index_pip_lm.x * processed_frame.shape[1]), int(index_pip_lm.y * processed_frame.shape[0]))
                    mcp_pos = (int(index_mcp_lm.x * processed_frame.shape[1]), int(index_mcp_lm.y * processed_frame.shape[0]))
                    
                    # Check if using thumbs up/down gesture for scrolling
                    if self.hand_detector.is_thumbs_up(hand_landmark.landmark) or self.hand_detector.is_thumbs_down(hand_landmark.landmark):
                        # Skip mouse movement when in scroll mode
                        pass
                    else:
                        # Move cursor when tracking is active, hand is detected, and not in scroll mode
                        new_x, new_y = self.mouse_controller.move_mouse_relative(index_tip.x, index_tip.y)
                        if new_x is not None:
                            # Check if gesture text was "Hand not detected" and clear it or set to tracking
                            if self.gestureOutput.text() == "Gesture: Hand not detected":
                                 self.gestureOutput.setText(f"Tracking: Index Finger")
                            # else, it might be showing a click gesture, so don't overwrite immediately unless it's the default
                            elif self.gestureOutput.text() != "Gesture: Left Click" and self.gestureOutput.text() != "Gesture: Right Click" and self.gestureOutput.text() != "Gesture: Scroll Up" and self.gestureOutput.text() != "Gesture: Scroll Down":
                                 self.gestureOutput.setText(f"Tracking: Index Finger")

                    # Add visual indicator for active tracking
                    cv2.circle(processed_frame, index_pos, 15, (0, 255, 0), -1)  # Green circle for active tracking
                    cv2.putText(processed_frame, "TRACKING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Visualize the finger joints by drawing lines
                    cv2.line(processed_frame, index_pos, dip_pos, (255, 255, 0), 2)
                    cv2.line(processed_frame, dip_pos, pip_pos, (255, 255, 0), 2)
                    cv2.line(processed_frame, pip_pos, mcp_pos, (255, 255, 0), 2)
                    
                    # Detect gestures (clicks and scrolls)
                    processed_frame, gesture = self.mouse_controller.detect_gestures(processed_frame, hand_landmark)
                    if gesture:
                        self.gestureOutput.setText(f"Gesture: {gesture}")
                
                else: # Hand is not detected, but tracking is ON
                    self.mouse_controller.reset_tracking() # Reset to prevent jumps when hand reappears
                    self.gestureOutput.setText("Gesture: Hand not detected") 
            
            # Convert frame to QImage for display
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_img = QImage(rgb_image.data, w, h, w * ch, QImage.Format.Format_RGB888)
            self.videoFeed.setPixmap(QPixmap.fromImage(qt_img))
            
            # Explicitly release OpenCV image to free memory
            processed_frame = None
            rgb_image = None

    # Calibration mouse event (commented out but kept for reference)
    """
    def mousePressEvent(self, event):
        if self.is_calibrating and event.button() == Qt.MouseButton.LeftButton:
            self.current_calibration_step += 1
            if self.current_calibration_step >= len(self.calibration_steps):
                self.is_calibrating = False
                self.gestureOutput.setText("Calibration: Completed")
                self.calibrationStatus.setText("Calibration: Completed")
            else:
                next_step = self.calibration_steps[self.current_calibration_step]
                self.gestureOutput.setText(f"Calibration: Position hand at {next_step} and click")
    """

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
        
    def closeEvent(self, event):
        """Clean up resources when the application closes"""
        self.stop_camera()
        # Force garbage collection to clean up memory
        self.force_garbage_collection()
        super().closeEvent(event)

# Helper class to track application state
class AppState:
    def __init__(self):
        self.transition_complete = False

if __name__ == '__main__':
    # Initialize application
    app = QApplication(sys.argv)
    
    # Create main window but don't show it yet
    main_app = AirFlick()
    main_app.setWindowTitle("AirFlick - Innovate the way you interact")
    
    # Get paths to images
    current_dir = os.path.dirname(os.path.abspath(__file__))
    splash_image_path = os.path.join(current_dir, "4537657.jpg")  # Keep original splash image
    logo_path = os.path.join(current_dir, "7b6148fa-18c6-4439-b089-be2a0dce14c9.png")  # Logo for welcome screen
    
    # Create app state to track transition
    app_state = AppState()
    
    def show_main_window():
        """Function to show main window and handle transition"""
        if app_state.transition_complete:
            return  # Prevent duplicate executions
            
        print("Showing main window")
        app_state.transition_complete = True
        
        # Center the main window
        screen_geometry = app.primaryScreen().geometry()
        x = (screen_geometry.width() - main_app.width()) // 2
        y = (screen_geometry.height() - main_app.height()) // 2
        main_app.move(x, y)
        
        # Show main window and bring to front
        main_app.show()
        main_app.raise_()
        main_app.activateWindow()
        
        # Force window manager to recognize and display window
        app.processEvents()
    
    # Create and show welcome screen - using splash image for welcome, the logo is already in the UI
    welcome = WelcomeScreen(splash_image_path)
    welcome.animation_finished.connect(show_main_window)
    welcome.centerOnScreen()
    welcome.show()
    
    # Start a safety timer to ensure main window shows even if animation signal fails
    safety_timer = QTimer()
    safety_timer.setSingleShot(True)
    safety_timer.timeout.connect(show_main_window)
    safety_timer.start(5000)  # 5 seconds max wait
    
    # Final garbage collection before main loop
    gc.collect()
    
    sys.exit(app.exec())