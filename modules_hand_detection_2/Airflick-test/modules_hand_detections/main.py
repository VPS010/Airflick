import sys
import cv2
import numpy as np
import os
import gc
from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QMainWindow, QMessageBox
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6 import uic

from hand_detection import HandDetector
from mouse_controller import MouseController
from welcome_screen import WelcomeScreen
from screenshot_controller import ScreenshotController
from camera_manager import CameraManager

class AirFlick(QWidget):
    def __init__(self):
        super().__init__()
        # Construct the path to air_flick.ui relative to main.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_path = os.path.join(current_dir, "air_flick.ui")
        
        # Verify the UI file exists
        if not os.path.exists(ui_path):
            QMessageBox.critical(None, "Error", f"UI file not found at: {ui_path}\nPlease ensure air_flick.ui is in the same directory as main.py.")
            raise FileNotFoundError(f"UI file not found at: {ui_path}")
        
        # Load the UI
        try:
            uic.loadUi(ui_path, self)
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load UI file: {str(e)}")
            raise
        
        self.hand_detector = HandDetector()
        self.mouse_controller = MouseController()
        self.mouse_controller.hand_detector = self.hand_detector  # Assign HandDetector to MouseController
        self.screenshot_controller = ScreenshotController()
        self.camera_manager = CameraManager(self.hand_detector, self.mouse_controller, self.screenshot_controller)
        self.camera_manager.frame_updated.connect(self.update_video_feed)
        self.camera_manager.status_text_updated.connect(self.gestureOutput.setText)
        self.camera_manager.cursor_position_updated.connect(self.positionOutput.setText)
        self.is_tracking = False
        self.is_calibrating = False
        self.startButton.clicked.connect(self.start_tracking)
        self.stopButton.clicked.connect(self.stop_all)
        self.settingsButton.clicked.connect(self.show_settings)
        self.sensitivitySlider.setMinimum(1)
        self.sensitivitySlider.setMaximum(10)
        self.sensitivitySlider.setValue(4)
        self.sensitivitySlider.valueChanged.connect(self.update_sensitivity)
        self.sensitivityValue.setText(f"{self.mouse_controller.scaling_factor:.1f}")
        self.smoothnessSlider.setMinimum(1)
        self.smoothnessSlider.setMaximum(10)
        self.smoothnessSlider.setValue(int(self.mouse_controller.smooth_factor * 10))
        self.smoothnessValue.setText(f"{self.mouse_controller.smooth_factor:.1f}")
        self.gc_timer = QTimer()
        self.gc_timer.timeout.connect(self.force_garbage_collection)
        self.gc_timer.start(60000)

    def start_tracking(self):
        self.camera_manager.set_tracking_status(True)
        self.camera_manager.start_feed()

    def stop_all(self):
        self.camera_manager.set_tracking_status(False)
        self.camera_manager.stop_feed()
        self.force_garbage_collection()

    def force_garbage_collection(self):
        gc.collect()

    def show_settings(self):
        self.gestureOutput.setText("Gesture: Settings button clicked")

    def update_video_feed(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        self.videoFeed.setPixmap(pixmap)

    def update_sensitivity(self, value):
        scaling_factor = float(value)
        self.mouse_controller.scaling_factor = scaling_factor
        self.sensitivityValue.setText(f"{scaling_factor:.1f}")

    def update_smoothness(self, value):
        smoothness = value / 10.0
        self.mouse_controller.smooth_factor = smoothness
        self.smoothnessValue.setText(f"{smoothness:.1f}")

    def closeEvent(self, event):
        self.camera_manager.stop_feed()
        self.force_garbage_collection()
        super().closeEvent(event)

class AppState:
    def __init__(self):
        self.transition_complete = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, "7b6148fa-18c6-4439-b089-be2a0dce14c9.png")
    
    # Verify the logo file exists
    if not os.path.exists(logo_path):
        QMessageBox.critical(None, "Error", f"Logo file not found at: {logo_path}\nPlease ensure the logo file is in the same directory as main.py.")
        raise FileNotFoundError(f"Logo file not found at: {logo_path}")
    
    main_app = AirFlick()
    main_app.setWindowTitle("AirFlick - Innovate the way you interact")
    app_state = AppState()
    
    def show_main_window():
        if app_state.transition_complete:
            return
        print("Showing main window")
        app_state.transition_complete = True
        screen_geometry = app.primaryScreen().geometry()
        x = (screen_geometry.width() - main_app.width()) // 2
        y = (screen_geometry.height() - main_app.height()) // 2
        main_app.move(x, y)
        main_app.show()
        main_app.raise_()
        main_app.activateWindow()
        app.processEvents()
    
    welcome = WelcomeScreen(logo_path)
    welcome.animation_finished.connect(show_main_window)
    welcome.centerOnScreen()
    welcome.show()
    safety_timer = QTimer()
    safety_timer.setSingleShot(True)
    safety_timer.timeout.connect(show_main_window)
    safety_timer.start(5000)
    gc.collect()
    sys.exit(app.exec())
