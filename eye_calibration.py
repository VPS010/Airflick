import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize webcam
cam = cv2.VideoCapture(0)

# Scroll parameters
max_scroll_speed = 90  # Maximum scroll step
min_scroll_speed = 20   # Minimum scroll step
scroll_delay = 0.01     # Shorter delay for smoother scrolling
neutral_zone = 0.05     # Buffer zone for no scrolling

# Sensitivity factor for scroll speed
sensitivity_factor = 1.5

# Eye landmark indices for MediaPipe Face Mesh
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

# Function to calculate the vertical position of the pupil normalized between 0 and 1
def calculate_normalized_position(eye_landmarks, window_h):
    top_of_eye = int(eye_landmarks[1].y * window_h)  # Top border of the eye
    bottom_of_eye = int(eye_landmarks[4].y * window_h)  # Bottom border of the eye
    pupil_y = int(eye_landmarks[0].y * window_h)  # Pupil's Y position

    # Normalize the pupil position
    if bottom_of_eye != top_of_eye:
        normalized_position = (pupil_y - top_of_eye) / (bottom_of_eye - top_of_eye)  # Scale to [0, 1]
    else:
        normalized_position = 0.5  # Default to center if division by zero occurs

    return normalized_position

# Moving average to smooth out the vertical position
eye_positions = []
def smooth_eye_position(new_position, window_size=5):
    eye_positions.append(new_position)
    if len(eye_positions) > window_size:
        eye_positions.pop(0)
    return sum(eye_positions) / len(eye_positions)

# Function to calculate dynamic scroll speed
def calculate_scroll_speed(relative_position, neutral_threshold, max_speed, min_speed):
    distance_from_neutral = abs(relative_position - neutral_threshold)
    normalized_distance = min(distance_from_neutral / 0.5, 1.0)  # Normalize the distance (0 to 1)
    dynamic_speed = min_speed + (max_speed - min_speed) * normalized_distance
    return dynamic_speed * sensitivity_factor

# Capture the current normalized eye position
def get_current_normalized_position():
    ret, image = cam.read()
    if not ret:
        print("Error: Could not read from camera.")
        return None

    window_h, _ = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = face_mesh.process(rgb_image)

    if processed_image.multi_face_landmarks:
        one_face_landmark_points = processed_image.multi_face_landmarks[0].landmark
        left_eye_landmarks = [one_face_landmark_points[idx] for idx in left_eye_indices]
        return calculate_normalized_position(left_eye_landmarks, window_h)
    else:
        print("Warning: Face landmarks not detected.")
        return None

# Calibration function to set personalized thresholds
def calibrate_user_thresholds():
    print("Calibration started: Please follow instructions.")
    
    # Step 1: Capture neutral (center) position
    input("Look straight ahead and press Enter.")
    neutral_position = get_current_normalized_position()
    if neutral_position is None:
        print("Error: Unable to capture neutral position. Please try again.")
        return None, None

    # Step 2: Capture upward position
    input("Look up as far as comfortable and press Enter.")
    up_position = get_current_normalized_position()
    if up_position is None:
        print("Error: Unable to capture upward position. Please try again.")
        return None, None

    # Step 3: Capture downward position
    input("Look down as far as comfortable and press Enter.")
    down_position = get_current_normalized_position()
    if down_position is None:
        print("Error: Unable to capture downward position. Please try again.")
        return None, None

    # Define personalized thresholds based on captured values
    personalized_scroll_up_threshold = neutral_position + (up_position - neutral_position) * 0.7
    personalized_scroll_down_threshold = neutral_position + (down_position - neutral_position) * 0.7
    
    print("Calibration complete. Thresholds set for this session.")
    
    return personalized_scroll_up_threshold, personalized_scroll_down_threshold

# Run calibration and get personalized thresholds
personalized_scroll_up_threshold, personalized_scroll_down_threshold = calibrate_user_thresholds()

# If calibration failed, exit the program
if personalized_scroll_up_threshold is None or personalized_scroll_down_threshold is None:
    print("Calibration failed. Exiting program.")
    cam.release()
    face_mesh.close()
    exit()

# Main loop
while True:
    ret, image = cam.read()

    if not ret:
        break

    # Get image dimensions
    window_h, window_w, _ = image.shape

    # Convert BGR image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect facial landmarks
    processed_image = face_mesh.process(rgb_image)
    all_faces_landmark_points = processed_image.multi_face_landmarks

    if all_faces_landmark_points:
        one_face_landmark_points = all_faces_landmark_points[0].landmark

        # Get vertical position of pupil in left eye
        left_eye_landmarks = [one_face_landmark_points[idx] for idx in left_eye_indices]
        normalized_position_left_eye = calculate_normalized_position(left_eye_landmarks, window_h)

        # Smooth the vertical position to reduce jitter
        smoothed_position = smooth_eye_position(normalized_position_left_eye)

        # Scroll based on the smoothed vertical eye position with personalized thresholds
        if smoothed_position < (personalized_scroll_up_threshold - neutral_zone):
            dynamic_scroll_speed = calculate_scroll_speed(smoothed_position, personalized_scroll_up_threshold, max_scroll_speed, min_scroll_speed)
            print(f"Looking Up - Smooth Scroll Up, Speed: {dynamic_scroll_speed}")
            pyautogui.scroll(-int(dynamic_scroll_speed))  # Scroll up with dynamic speed
            time.sleep(scroll_delay)
        elif smoothed_position > (personalized_scroll_down_threshold + neutral_zone):
            dynamic_scroll_speed = calculate_scroll_speed(smoothed_position, personalized_scroll_down_threshold, max_scroll_speed, min_scroll_speed)
            print(f"Looking Down - Smooth Scroll Down, Speed: {dynamic_scroll_speed}")
            pyautogui.scroll(int(dynamic_scroll_speed))  # Scroll down with dynamic speed
            time.sleep(scroll_delay)
        else:
            print("Neutral Zone - No Scrolling")  # No scrolling in the neutral zone

        # Draw circles on the left and right eye landmarks
        for idx in left_eye_indices:
            x = int(one_face_landmark_points[idx].x * window_w)
            y = int(one_face_landmark_points[idx].y * window_h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        for idx in right_eye_indices:
            x = int(one_face_landmark_points[idx].x * window_w)
            y = int(one_face_landmark_points[idx].y * window_h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # Show the video stream with eye landmarks
    cv2.imshow("Eye Detection", image)

    # Break the loop on 'ESC' key press
    key = cv2.waitKey(1)
    if key == 27:  # Escape key
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
