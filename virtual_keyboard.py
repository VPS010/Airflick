import sys
import cv2
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, QSize, QPoint, QTimer
from PyQt6.QtGui import QFont, QGuiApplication
import pyautogui
import time

class VirtualKeyboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setStyleSheet("background-color: #1e293b; border-radius: 10px; border: 1px solid #334155;")
        self.init_ui()
        self.active_input = None
        self.shift_active = False
        self.caps_lock = False
        self.is_dragging = False
        self.drag_position = QPoint()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        # Keyboard title
        title = QLabel("AirFlick Virtual Keyboard")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8; border: none;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Keyboard rows
        keys = [
            ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "Backspace"],
            ["Tab", "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[", "]", "\\"],
            ["Caps", "a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "'", "Enter"],
            ["Shift", "z", "x", "c", "v", "b", "n", "m", ",", ".", "/", "Shift"],
            ["Space"]
        ]

        for row in keys:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(5)
            for key in row:
                btn = QPushButton(key)
                btn.setFixedSize(50, 50) if key not in ["Backspace", "Enter", "Shift", "Space", "Tab", "Caps"] else btn.setFixedSize(80, 50)
                if key == "Space":
                    btn.setFixedSize(300, 50)
                btn.setStyleSheet("background-color: #334155; color: #e2e8f0; border-radius: 5px; border: 1px solid #475569; font-size: 14px;")
                btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                btn.clicked.connect(lambda checked, k=key: self.key_pressed(k))
                row_layout.addWidget(btn)
            layout.addLayout(row_layout)

        # Close button
        close_btn = QPushButton("Close Keyboard")
        close_btn.setFixedHeight(40)
        close_btn.setStyleSheet("background-color: #f87171; color: #ffffff; border-radius: 5px; font-weight: bold;")
        close_btn.clicked.connect(self.hide)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.drag_position = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            event.accept()

    def key_pressed(self, key):
        if key == "Backspace":
            pyautogui.press('backspace')
        elif key == "Enter":
            pyautogui.press('enter')
        elif key == "Shift":
            self.shift_active = not self.shift_active
            self.update_key_display()
        elif key == "Caps":
            self.caps_lock = not self.caps_lock
            self.update_key_display()
        elif key == "Tab":
            pyautogui.press('tab')
        elif key == "Space":
            pyautogui.press('space')
        else:
            char = key
            if len(char) == 1:
                if self.shift_active or self.caps_lock:
                    char = char.upper()
                else:
                    char = char.lower()
                if self.shift_active:
                    self.shift_active = False
                    self.update_key_display()
                pyautogui.write(char)

        # Ensure focus remains on the input field if one was active, using a small delay
        if self.active_input:
            QTimer.singleShot(10, lambda: self.active_input.setFocus())

    def update_key_display(self):
        for btn in self.findChildren(QPushButton):
            text = btn.text()
            if len(text) == 1 and text.isalpha():
                btn.setText(text.upper() if (self.shift_active or self.caps_lock) else text.lower())
            elif text == "Shift":
                btn.setStyleSheet("background-color: #38bdf8; color: #ffffff; border-radius: 5px; border: 1px solid #475569; font-size: 14px;" if self.shift_active else "background-color: #334155; color: #e2e8f0; border-radius: 5px; border: 1px solid #475569; font-size: 14px;")
            elif text == "Caps":
                btn.setStyleSheet("background-color: #38bdf8; color: #ffffff; border-radius: 5px; border: 1px solid #475569; font-size: 14px;" if self.caps_lock else "background-color: #334155; color: #e2e8f0; border-radius: 5px; border: 1px solid #475569; font-size: 14px;")

    def set_active_input(self, input_widget):
        self.active_input = input_widget
        if input_widget:

            self.show()
        else:
            self.hide()

    def show(self):
        super().show()
        # Position keyboard at bottom center of the primary screen
        screen = QGuiApplication.primaryScreen()
        geometry = screen.availableGeometry() if screen else self.screen().availableGeometry()
        x = (geometry.width() - self.width()) // 2
        y = geometry.height() - self.height() - 180  # 180 px above bottom
        self.move(x, y)
        self.raise_()

    def hideEvent(self, event):
        self.active_input = None
        super().hideEvent(event)
