# AirFlick - Gesture-Based Mouse Control

## Overview
AirFlick is an innovative application that transforms hand gestures into mouse control using computer vision. With AirFlick, you can interact with your computer through intuitive hand movements, providing a touchless and futuristic user experience.

## Features
- **Gesture-Based Mouse Control**: Move the cursor, click, scroll, and perform other mouse actions using hand gestures.
- **Customizable Settings**: Adjust sensitivity, smoothness, and scroll speed to tailor the control experience to your preference.
- **Lighting Filters**: Enhance gesture detection in various lighting conditions with low light and high light compensation filters.
- **Intuitive UI**: User-friendly interface built with PyQt6 for seamless interaction.

## How It Works
AirFlick uses a webcam to capture hand movements, processes them using advanced computer vision algorithms, and translates them into mouse actions. The core components include:

- **Hand Detection**: Utilizes the `HandDetector` class to identify and track hand landmarks.
- **Mouse Controller**: The `MouseController` class maps hand gestures to mouse movements and actions.
- **UI Components**: Built with PyQt6, providing video feedback and settings customization.

## Installation
1. Clone this repository to your local machine.
2. Ensure you have Python 3.10 installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python main.py
   ```

## Requirements
See `requirements.txt` for the full list of dependencies. Key libraries include:
- OpenCV (`cv2`) for video capture and processing
- PyQt6 for the graphical user interface
- NumPy for numerical operations

## Usage
- Launch the application and allow webcam access.
- Use the 'Start Tracking' button to begin gesture control.
- Move your index finger to control the cursor.
- Perform gestures for clicking, scrolling, and other actions as detailed below.
- Adjust settings via sliders for a personalized experience.
- Use 'Stop' to pause gesture tracking.

## Controls and Settings
Airflick provides several adjustable settings to fine-tune your gesture control experience:
- **Sensitivity Slider**: Adjusts how responsive the cursor is to hand movements. Higher values make the cursor move faster with smaller hand movements (Range: 2 to 8).
- **Smoothness Slider**: Controls the smoothing effect on cursor movement to reduce jitter. Higher values result in smoother but potentially slower response (Range: 0.1 to 1.0).
- **Scroll Speed Slider**: Determines the speed of scrolling when using scroll gestures (Range: 0.1 to 2.0).

## Filters for Optimal Detection
Airflick includes lighting filters to improve hand detection under various conditions:
- **Low Light Filter**: Enhances visibility in dim environments by adjusting the image brightness. Enable this when working in poorly lit areas to improve gesture recognition.
- **High Light Filter**: Compensates for overly bright environments by adjusting gamma levels. Use this in bright settings to reduce glare and improve detection accuracy. Note: Only one filter can be active at a time.

## Gestures
Airflick translates specific hand gestures into mouse actions for intuitive control. Below are the primary gestures supported:
- **Cursor Movement**: Extend only your index finger and move your hand to control the cursor position on screen. Other fingers should be folded down.
- **Left Click**: Form a pinching gesture with thumb and index finger (distance less than 0.04 normalized units) to simulate a left mouse click.
- **Right Click**: Form a pinching gesture with thumb and middle finger (distance less than 0.04 normalized units) for a right mouse click.
- **Scroll Up**: Perform a thumbs-up gesture with thumb extended upward and other fingers folded to scroll up. The speed depends on the scroll speed setting.
- **Scroll Down**: Perform a thumbs-down gesture with thumb extended downward and other fingers folded to scroll down. The speed depends on the scroll speed setting.
- **Screenshot Trigger**: Bring all five fingertips close together (pairwise distance less than 0.1 normalized units) to trigger a screenshot. This activates the OS screenshot tool (PrintScreen on Windows/Linux).

Additional gestures for advanced controls are being developed and will be documented as they are implemented in `mouse_controller.py`.

## Project Structure
- `main.py`: Main application file that integrates all components.
- `hand_detection.py`: Contains the logic for detecting and tracking hand landmarks.
- `mouse_controller.py`: Handles the conversion of hand movements to mouse actions.
- `welcome_screen.py`: Displays the initial splash screen.
- `screenshot_trigger.py`: Manages screenshot functionality via gestures.
- `air_flick.ui`: UI definition file for the PyQt6 interface.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for bugs, features, or improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details (if applicable, or update as per your licensing choice).

## Acknowledgments
AirFlick is built leveraging powerful open-source libraries like OpenCV and PyQt6, and inspired by the potential of touchless interfaces in modern computing.

Innovate the way you interact with AirFlick!
