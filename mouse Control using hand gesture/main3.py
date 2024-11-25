#corner calibration
#Calibration only movement
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe Hand Solutions
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Calibration points
calibration_corners = {
    "top_left": None,
    "top_right": None,
    "bottom_left": None,
    "bottom_right": None
}

# Smoothing factors
prev_x, prev_y = 0, 0
smooth_factor = 0.2


def get_finger_midpoint(landmarks, index, middle):
    """Calculate the midpoint of the index and middle finger tips."""
    x = (landmarks[index].x + landmarks[middle].x) / 2
    y = (landmarks[index].y + landmarks[middle].y) / 2
    return x, y


def move_mouse(x, y):
    """Move mouse smoothly to the given coordinates."""
    global prev_x, prev_y

    # Smoothing cursor movement
    x = int(prev_x + (x - prev_x) * smooth_factor)
    y = int(prev_y + (y - prev_y) * smooth_factor)

    try:
        pyautogui.moveTo(x, y)
        prev_x, prev_y = x, y
    except pyautogui.FailSafeException:
        print("PyAutoGUI fail-safe triggered. Move your mouse away from the corner.")


def calibrate_corners():
    """Calibrate the four corners of the screen."""
    print("Starting calibration. Please position your hand in the following corners.")
    cap = cv2.VideoCapture(0)
    corner_names = ["top_left", "top_right", "bottom_left", "bottom_right"]

    for corner in corner_names:
        print(f"Position your hand in the {corner.replace('_', ' ')} corner and press 'c'.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    cv2.putText(
                        frame,
                        f"Calibrating {corner}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

            cv2.imshow("Calibration", frame)

            # Wait for 'c' key to confirm corner calibration
            if cv2.waitKey(1) & 0xFF == ord('c'):
                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    calibration_corners[corner] = (
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    )
                    print(f"{corner.replace('_', ' ').capitalize()} calibrated at: {calibration_corners[corner]}")
                    break

    cap.release()
    cv2.destroyAllWindows()


def map_to_screen(x, y):
    """Map hand coordinates to screen coordinates based on calibrated corners."""
    top_left = calibration_corners["top_left"]
    top_right = calibration_corners["top_right"]
    bottom_left = calibration_corners["bottom_left"]

    # Map x and y from hand detection space to screen space
    screen_x = np.interp(x, [top_left[0], top_right[0]], [0, screen_width])
    screen_y = np.interp(y, [top_left[1], bottom_left[1]], [0, screen_height])
    return int(screen_x), int(screen_y)


def main():
    """Main function for controlling the cursor with hand gestures."""
    global prev_x, prev_y

    # Calibrate the corners before starting
    calibrate_corners()
    print("Calibration completed. Starting cursor control...")

    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get the midpoint of the index and middle finger tips
                    midpoint = get_finger_midpoint(
                        hand_landmarks.landmark,
                        mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                    )

                    # Map to screen coordinates
                    screen_x, screen_y = map_to_screen(midpoint[0], midpoint[1])

                    # Move the mouse
                    move_mouse(screen_x, screen_y)

            cv2.imshow("Hand Tracking", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
