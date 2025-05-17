import sys
import os
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QApplication, 
    QGraphicsOpacityEffect  # Imported from QtWidgets, not QtCore
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QSize, QRect
from PyQt6.QtGui import QPixmap, QMovie, QColor, QPainter, QPainterPath
from PyQt6 import QtCore

class LogoWidget(QLabel):
    """Custom label widget for logo with rounded corners"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Increased minimum size from 300x300 to 500x500
        self.setMinimumSize(500, 500)
        self.radius = 30  # Increased corner radius to match larger size
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Create rounded rect path
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), self.radius, self.radius)
        
        # Clip to the rounded rectangle
        painter.setClipPath(path)
        
        # Draw the pixmap within the clipping path
        super().paintEvent(event)


class WelcomeScreen(QWidget):
    """Welcome screen with animated logo display"""
    # Signal to emit when animation is complete
    animation_finished = QtCore.pyqtSignal()
    
    def __init__(self, logo_path):
        super().__init__()
        
        # Flag to track if we've already emitted the signal
        self._transition_started = False
        
        # Set window properties
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Increased the window size to 1000x800 to accommodate larger logo
        self.resize(1000, 800)
        
        # Create layout
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # App title
        self.title_label = QLabel("AirFlick", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            font-family: 'SF Pro Display', 'Arial';
            font-size: 60px;  /* Increased font size */
            font-weight: bold;
            color: #38bdf8;
            margin-bottom: 30px;  /* Increased margin */
        """)
        self.title_label.setVisible(False)  # Initially hidden
        
        # Logo widget
        self.logo_widget = LogoWidget(self)
        self.logo_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_widget.setScaledContents(True)
        
        # Load the logo
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Scale pixmap while maintaining aspect ratio - increased from 300x300 to 500x500
            pixmap = pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio, 
                                  Qt.TransformationMode.SmoothTransformation)
            self.logo_widget.setPixmap(pixmap)
        else:
            # Fallback message if logo is not found
            self.logo_widget.setText("Logo Not Found")
            self.logo_widget.setStyleSheet("background-color: #0f172a; color: white; border-radius: 30px;")
        
        # Set initial opacity to 0
        self.logo_widget.setGraphicsEffect(None)  # Clear any existing effects
        self.logo_opacity_effect = QGraphicsOpacityEffect(self.logo_widget)
        self.logo_opacity_effect.setOpacity(0.0)
        self.logo_widget.setGraphicsEffect(self.logo_opacity_effect)
        
        # Subtitle with new tagline
        self.subtitle_label = QLabel("Innovate the way you interact", self)
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setStyleSheet("""
            font-family: 'SF Pro Display', 'Arial';
            font-size: 28px;  /* Increased font size */
            color: #94a3b8;
            margin-top: 20px;  /* Increased margin */
        """)
        self.subtitle_label.setVisible(False)  # Initially hidden
        
        # Add widgets to layout
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.logo_widget)
        self.layout.addWidget(self.subtitle_label)
        
        # Style the window with new color scheme
        self.setStyleSheet("""
            WelcomeScreen {
                background-color: #0f172a;
            }
        """)
        
        # Setup animation sequence
        self.setup_animations()
        
        # Start animations after a short delay
        QTimer.singleShot(300, self.start_animations)  # Increased initial delay
        
        # Auto-close after animation
        self.close_timer = QTimer(self)
        self.close_timer.setSingleShot(True)
        self.close_timer.timeout.connect(self.finish_animation)
        
        # Set a safety timer to ensure we emit the signal even if animations fail
        self.safety_timer = QTimer(self)
        self.safety_timer.setSingleShot(True)
        self.safety_timer.timeout.connect(self.finish_animation)
        self.safety_timer.start(1200)  # Increased to 6 seconds max timeout

    def setup_animations(self):
        # Logo fade-in animation - increased duration
        self.logo_fade_in = QPropertyAnimation(self.logo_opacity_effect, b"opacity")
        self.logo_fade_in.setDuration(1200)  # Increased from 800ms to 1200ms
        self.logo_fade_in.setStartValue(0.0)
        self.logo_fade_in.setEndValue(1.0)
        self.logo_fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Logo bounce animation (scales up then down slightly)
        self.logo_bounce = QPropertyAnimation(self.logo_widget, b"geometry")
        rect = self.logo_widget.geometry()
        center_x = rect.x() + rect.width() / 2
        center_y = rect.y() + rect.height() / 2
        
        self.logo_bounce.setDuration(800)  # Increased from 500ms to 800ms
        # Start with current size
        self.logo_bounce.setStartValue(rect)
        # Scale up slightly - using slightly larger scale factor (1.15 instead of 1.1)
        bigger_rect = QRect(
            center_x - (rect.width() * 1.15) / 2,
            center_y - (rect.height() * 1.15) / 2,
            rect.width() * 1.15,
            rect.height() * 1.15
        )
        self.logo_bounce.setEndValue(bigger_rect)
        self.logo_bounce.setEasingCurve(QEasingCurve.Type.OutBack)
        
        # Connect animations in sequence
        self.logo_fade_in.finished.connect(self.show_title)

    def start_animations(self):
        print("Starting welcome animations")
        self.logo_fade_in.start()

    def show_title(self):
        # Show the title with fade-in
        self.title_label.setVisible(True)
        
        # Create and configure opacity effect
        title_opacity = QGraphicsOpacityEffect(self.title_label)
        title_opacity.setOpacity(0.0)
        self.title_label.setGraphicsEffect(title_opacity)
        
        # Create animation
        self.title_anim = QPropertyAnimation(title_opacity, b"opacity")
        self.title_anim.setDuration(800)  # Increased from 500ms to 800ms
        self.title_anim.setStartValue(0.0)
        self.title_anim.setEndValue(1.0)
        
        # Start animation and connect to next step
        self.title_anim.finished.connect(self.show_subtitle)
        self.title_anim.start()
        
        # Start logo bounce animation
        self.logo_bounce.start()

    def show_subtitle(self):
        # Show subtitle with fade-in
        self.subtitle_label.setVisible(True)
        
        # Create and configure opacity effect
        subtitle_opacity = QGraphicsOpacityEffect(self.subtitle_label)
        subtitle_opacity.setOpacity(0.0)
        self.subtitle_label.setGraphicsEffect(subtitle_opacity)
        
        # Create animation
        self.subtitle_anim = QPropertyAnimation(subtitle_opacity, b"opacity")
        self.subtitle_anim.setDuration(800)  # Increased from 500ms to 800ms
        self.subtitle_anim.setStartValue(0.0)
        self.subtitle_anim.setEndValue(1.0)
        
        # Start animation and set timer to close
        self.subtitle_anim.start()
        
        # Schedule close after animation completes - increased to 2000ms (2 seconds) of visibility
        self.close_timer.start(2000)
        print("Welcome animations running, scheduled transition in 2 seconds")

    def finish_animation(self):
        # Only emit the signal once
        if not self._transition_started:
            self._transition_started = True
            print("Welcome screen animation completed, emitting finished signal")
            
            # Cancel the safety timer if normal timer triggered this
            if self.safety_timer.isActive():
                self.safety_timer.stop()
                
            # Emit signal that animation is complete
            self.animation_finished.emit()
            
            # Add a small delay before hiding ourselves
            QTimer.singleShot(100, self.hide_welcome)

    def hide_welcome(self):
        """Properly hide the welcome screen to ensure main window appears"""
        print("Hiding welcome screen")
        self.hide()
        # Force this window to close and be deleted
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.close()

    def centerOnScreen(self):
        """Center the window on the screen"""
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2),
                  int((screen.height() - size.height()) / 2))


if __name__ == "__main__":
    # For testing purposes
    app = QApplication(sys.argv)
    logo_path = "4537657.jpg"  # Adjust path as needed
    welcome = WelcomeScreen(logo_path)
    welcome.centerOnScreen()
    welcome.show()
    sys.exit(app.exec()) 