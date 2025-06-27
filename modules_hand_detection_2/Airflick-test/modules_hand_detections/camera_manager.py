import cv2
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QImage

class CameraManager(QObject):
    frame_updated = pyqtSignal(QImage)
    gesture_detected = pyqtSignal(str)
    tracking_status_updated = pyqtSignal(str)
    status_text_updated = pyqtSignal(str)  # Unified status text signal
    cursor_position_updated = pyqtSignal(str)  # For cursor position display

    def __init__(self, hand_detector, mouse_controller, screenshot_controller, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_frame)
        self.hand_detector = hand_detector
        self.mouse_controller = mouse_controller
        self.screenshot_controller = screenshot_controller
        self.is_tracking_active = False

    def start_feed(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        if self.cap and not self.cap.isOpened():
            print("Error: Could not open video stream.")
            self.cap = None
            self.tracking_status_updated.emit("Error: Camera not found")
            return False
        if not self.timer.isActive():
            self.timer.start(30)
        return True

    def stop_feed(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def set_tracking_status(self, is_tracking):
        self.is_tracking_active = is_tracking

    def _process_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        processed_frame, hand_landmarks_list = self.hand_detector.find_hands(frame.copy())
        status_text = "Status: Idle"
        if self.is_tracking_active:
            if hand_landmarks_list:
                hand_landmarks = hand_landmarks_list[0]
                index_tip = hand_landmarks.landmark[self.hand_detector.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                new_x, new_y = self.mouse_controller.move_mouse_relative(index_tip.x, index_tip.y)
                if new_x is not None:
                    self.cursor_position_updated.emit(f"X: {int(new_x)}, Y: {int(new_y)}")
                index_pos_viz = (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]))
                cv2.circle(processed_frame, index_pos_viz, 15, (0, 255, 0), -1)
                cv2.putText(processed_frame, "TRACKING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Detect mouse gestures
                _, mouse_gesture = self.mouse_controller.detect_gestures(frame, hand_landmarks)
                if mouse_gesture:
                    status_text = f"Gesture: {mouse_gesture}"
                else:
                    status_text = "Tracking: Index Finger"
                # Detect screenshot gesture
                processed_frame, screenshot_gesture = self.screenshot_controller.detect_screenshot_gesture(processed_frame, hand_landmarks)
                if screenshot_gesture:
                    status_text = f"Gesture: {screenshot_gesture}"
            else:
                self.mouse_controller.reset_tracking()
                status_text = "Gesture: Hand not detected"
        else:
            if hand_landmarks_list:
                status_text = "Status: Hand Detected (Tracking Off)"
            else:
                status_text = "Status: No Hand Detected (Tracking Off)"
        self.status_text_updated.emit(status_text)
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.frame_updated.emit(qt_img)

    def __del__(self):
        self.stop_feed()