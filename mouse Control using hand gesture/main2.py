import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Set up the mouse and screen parameters
screen_width, screen_height = pyautogui.size()
scaling_factor = 1.5  # Scaling factor for cursor speed
cursor_smooth_factor = 0.6  # Smoothing factor for cursor movement
sensitivity_factor = 0.5  # Sensitivity for finger distance-based movement
max_x, max_y = screen_width * 0.8, screen_height * 0.8  # Limit movement to 80% of the screen

# Initialize Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
draw = mp.solutions.drawing_utils

# For stabilizing movement
prev_x, prev_y = None, None

def get_midpoint(finger1, finger2):
    return [(finger1.x + finger2.x) / 2, (finger1.y + finger2.y) / 2]

def calculate_distance(finger1, finger2):
    return np.hypot(finger2.x - finger1.x, finger2.y - finger1.y) * sensitivity_factor

def smooth_movement(new_x, new_y):
    global prev_x, prev_y
    if prev_x is None or prev_y is None:
        prev_x, prev_y = new_x, new_y

    # Apply smoothing
    smoothed_x = prev_x + (new_x - prev_x) * cursor_smooth_factor
    smoothed_y = prev_y + (new_y - prev_y) * cursor_smooth_factor
    prev_x, prev_y = smoothed_x, smoothed_y

    # Apply screen boundaries
    smoothed_x = min(max(smoothed_x, 0), max_x)
    smoothed_y = min(max(smoothed_y, 0), max_y)

    # Scaling factor for speed adjustment
    smoothed_x *= scaling_factor
    smoothed_y *= scaling_factor

    return smoothed_x, smoothed_y

def move_mouse(midpoint):
    if midpoint:
        x, y = int(midpoint[0] * screen_width), int(midpoint[1] * screen_height)
        smoothed_x, smoothed_y = smooth_movement(x, y)
        pyautogui.moveTo(smoothed_x, smoothed_y)

def main():
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frame_rgb)

            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the index and middle fingertip landmarks
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Calculate the midpoint for more stable cursor movement
                midpoint = get_midpoint(index_tip, middle_tip)
                
                # Move the cursor based on midpoint with stabilization
                move_mouse(midpoint)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
