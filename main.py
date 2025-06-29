import sys
import cv2
import numpy as np
import os
import gc  # Import garbage collector
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import QTimer, QObject, QEvent
from PyQt6.QtGui import QImage, QPixmap
from PyQt6 import uic

# Import our custom modules
from hand_detection import HandDetector
from mouse_controller import MouseController
from welcome_screen import WelcomeScreen
from screenshot_trigger import ScreenshotTrigger
from virtual_keyboard import VirtualKeyboard

class AirFlick(QWidget):
    def __init__(self):
        super().__init__()
        
        uic.loadUi('air_flick.ui', self)
        
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.hand_detector = HandDetector()
        self.mouse_controller = MouseController()
        self.screenshot_trigger = ScreenshotTrigger(self.hand_detector)
        self.virtual_keyboard = VirtualKeyboard()
        self.virtual_keyboard_enabled = False
        
        self.mouse_controller.hand_detector = self.hand_detector
        
        self.is_tracking = False
        
        self.startButton.clicked.connect(self.start_tracking)
        self.stopButton.clicked.connect(self.stop_all)
        self.settingsButton.clicked.connect(self.show_settings)
        
        self.sensitivitySlider.setMinimum(2)
        self.sensitivitySlider.setMaximum(8)
        self.mouse_controller.scaling_factor = 4.0
        self.sensitivitySlider.setValue(int(self.mouse_controller.scaling_factor))
        self.sensitivitySlider.valueChanged.connect(self.update_sensitivity)
        self.sensitivityValue.setText(f"{self.mouse_controller.scaling_factor:.1f}")
        
        self.smoothnessSlider.setMinimum(1)
        self.smoothnessSlider.setMaximum(10)
        self.smoothnessSlider.setValue(int(self.mouse_controller.smooth_factor * 10))
        self.smoothnessSlider.valueChanged.connect(self.update_smoothness)
        self.smoothnessValue.setText(f"{self.mouse_controller.smooth_factor:.1f}")

        # Scroll speed slider
        self.scrollSpeedSlider.setMinimum(1)
        self.scrollSpeedSlider.setMaximum(20)
        self.scrollSpeedSlider.setValue(int(self.mouse_controller.scroll_speed_factor * 10))
        self.scrollSpeedSlider.valueChanged.connect(self.update_scroll_speed)
        self.scrollSpeedValue.setText(f"{self.mouse_controller.scroll_speed_factor:.1f}")

        # Low light enhancement toggle
        self.low_light_filter_enabled = False
        self.lowLightToggle.toggled.connect(self.toggle_low_light_filter)
        self.lowLightToggle.setChecked(self.low_light_filter_enabled)

        # High light compensation toggle
        self.high_light_filter_enabled = False
        self.highLightToggle.toggled.connect(self.toggle_high_light_filter)
        self.highLightToggle.setChecked(self.high_light_filter_enabled)

        # Virtual keyboard toggle
        self.virtualKeyboardToggle.toggled.connect(self.toggle_virtual_keyboard)
        self.virtualKeyboardToggle.setChecked(self.virtual_keyboard_enabled)
        
        # Install event filter at startup if virtual keyboard is enabled
        if self.virtual_keyboard_enabled:
            app = QApplication.instance()
            app.installEventFilter(self)
        
        self.gc_timer = QTimer()
        self.gc_timer.timeout.connect(self.force_garbage_collection)
        self.gc_timer.start(60000)
        
    def toggle_virtual_keyboard(self, checked):
        self.virtual_keyboard_enabled = checked
        status = "ON" if checked else "OFF"
        self.gestureOutput.setText(f"Virtual Keyboard: {status}")
        if checked:
            self.virtual_keyboard.show()
        elif self.virtual_keyboard.isVisible():
            self.virtual_keyboard.hide()
            self.virtual_keyboard.active_input = None

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
        self.start_camera()
        self.mouse_controller.reset_tracking()
        self.is_tracking = True
        self.gestureOutput.setText("Gesture: Tracking started")

    def stop_all(self):
        self.stop_camera()
        self.mouse_controller.reset_tracking()
        self.is_tracking = False
        self.gestureOutput.setText("Gesture: None")
        self.force_garbage_collection()

    def force_garbage_collection(self):
        gc.collect()
        
    def show_settings(self):
        self.gestureOutput.setText("Gesture: Settings button clicked")

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.FocusIn and self.virtual_keyboard_enabled:
            if hasattr(obj, 'text') and hasattr(obj, 'setText'):
                self.virtual_keyboard.set_active_input(obj)
        elif event.type() == QEvent.Type.FocusOut:
            if self.virtual_keyboard.active_input == obj:
                self.virtual_keyboard.set_active_input(None)
        return super().eventFilter(obj, event)

    def preprocess_for_hand_detection(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge([cl, a, b])
        processed_frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return processed_frame

    def preprocess_for_high_light(self, frame, gamma=0.75):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(frame, table)

    def update_frame(self):
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame.copy(), 1)
        
        if self.low_light_filter_enabled:
            processed_frame = self.preprocess_for_hand_detection(frame)
        elif self.high_light_filter_enabled:
            processed_frame = self.preprocess_for_high_light(frame)
        else:
            processed_frame = frame
        
        processed_frame, hand_landmarks = self.hand_detector.find_hands(processed_frame)
        
        if self.is_tracking and hand_landmarks:
            hand_landmark = hand_landmarks[0]
            
            if not (self.hand_detector.is_thumbs_up(hand_landmark.landmark) or self.hand_detector.is_thumbs_down(hand_landmark.landmark)):
                if self.screenshot_trigger.check_and_trigger(hand_landmark.landmark):
                    cv2.rectangle(processed_frame, (10, 10), (180, 60), (0, 200, 0), -1)
                    cv2.putText(processed_frame, 'SCREENSHOT', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)

                if self.hand_detector.is_index_finger_straight(hand_landmark.landmark):
                    index_tip = hand_landmark.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    self.mouse_controller.move_mouse_relative(index_tip.x, index_tip.y)
                    self.gestureOutput.setText("Tracking: Index Finger")
                    index_pos = (int(index_tip.x * processed_frame.shape[1]), int(index_tip.y * processed_frame.shape[0]))
                    cv2.circle(processed_frame, index_pos, 15, (0, 255, 0), -1)
                else:
                    self.mouse_controller.reset_tracking()
                    self.gestureOutput.setText("Gesture: Index Finger Folded")
            
            processed_frame, gesture = self.mouse_controller.detect_gestures(processed_frame, hand_landmark)
            if gesture:
                self.gestureOutput.setText(f"Gesture: {gesture}")
        elif self.is_tracking:
            self.mouse_controller.reset_tracking()
            self.gestureOutput.setText("Gesture: Hand not detected")
        
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, w * ch, QImage.Format.Format_RGB888)
        self.videoFeed.setPixmap(QPixmap.fromImage(qt_img))

    def update_sensitivity(self, value):
        scaling_factor = float(value)
        self.mouse_controller.scaling_factor = scaling_factor
        self.sensitivityValue.setText(f"{scaling_factor:.1f}")

    def update_smoothness(self, value):
        smoothness = value / 10.0
        self.mouse_controller.smooth_factor = smoothness
        self.smoothnessValue.setText(f"{smoothness:.1f}")

    def update_scroll_speed(self, value):
        # Scale the slider value to a more granular speed factor
        speed_factor = value / 10.0
        self.mouse_controller.set_scroll_speed(speed_factor)
        self.scrollSpeedValue.setText(f"{speed_factor:.1f}")

    def toggle_low_light_filter(self, checked):
        self.low_light_filter_enabled = checked
        if checked and self.high_light_filter_enabled:
            self.high_light_filter_enabled = False
            self.highLightToggle.setChecked(False)
        status = "ON" if checked else "OFF"
        self.gestureOutput.setText(f"Low Light Filter: {status}")

    def toggle_high_light_filter(self, checked):
        self.high_light_filter_enabled = checked
        if checked and self.low_light_filter_enabled:
            self.low_light_filter_enabled = False
            self.lowLightToggle.setChecked(False)
        status = "ON" if checked else "OFF"
        self.gestureOutput.setText(f"High Light Filter: {status}")
        
    def closeEvent(self, event):
        self.stop_camera()
        self.force_garbage_collection()
        super().closeEvent(event)

class AppState:
    def __init__(self):
        self.transition_complete = False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    main_app = AirFlick()
    main_app.setWindowTitle("AirFlick - Innovate the way you interact")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    splash_image_path = os.path.join(current_dir, "7b6148fa-18c6-4439-b089-be2a0dce14c9.png")
    
    app_state = AppState()
    
    def show_main_window():
        if app_state.transition_complete:
            return
            
        app_state.transition_complete = True
        
        screen_geometry = app.primaryScreen().geometry()
        x = (screen_geometry.width() - main_app.width()) // 2
        y = (screen_geometry.height() - main_app.height()) // 2
        main_app.move(x, y)
        
        main_app.show()
        main_app.raise_()
        main_app.activateWindow()
        
        app.processEvents()
    
    welcome = WelcomeScreen(splash_image_path)
    welcome.animation_finished.connect(show_main_window)
    welcome.centerOnScreen()
    welcome.show()
    
    safety_timer = QTimer()
    safety_timer.setSingleShot(True)
    safety_timer.timeout.connect(show_main_window)
    safety_timer.start(5000)
    
    gc.collect()
    
    sys.exit(app.exec())