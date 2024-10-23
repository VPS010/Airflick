import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize webcam
cam = cv2.VideoCapture(0)

# Set the scroll speed (higher for faster smoothness)
scroll_step = 25  # Larger value for faster scrolling
scroll_delay = 0.01  # Shorter delay for smoother, faster scroll

# Function to calculate the vertical position of the eyeball (pupil) relative to the eye's top and bottom borders
def calculate_vertical_position(eye_landmarks, window_h):
    top_of_eye = int(eye_landmarks[1].y * window_h)  # Top border of the eye
    bottom_of_eye = int(eye_landmarks[4].y * window_h)  # Bottom border of the eye
    pupil_y = int(eye_landmarks[0].y * window_h)  # Pupil's Y position (approximated by the center landmark)

    # Ensure no division by zero
    if bottom_of_eye != top_of_eye:
        # Calculate relative vertical position of the pupil (0 = top, 1 = bottom)
        relative_position = (pupil_y - top_of_eye) / (bottom_of_eye - top_of_eye)
    else:
        # Default to center if division by zero occurs
        relative_position = 0.5

    return relative_position

while True:
    ret, image = cam.read()

    if not ret:
        break

    # Get image dimensions (height first, then width)
    window_h, window_w, _ = image.shape

    # Convert BGR image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect facial landmarks
    processed_image = face_mesh.process(rgb_image)
    all_faces_landmark_points = processed_image.multi_face_landmarks

    if all_faces_landmark_points:
        one_face_landmark_points = all_faces_landmark_points[0].landmark

        # Eye landmark indices for MediaPipe Face Mesh
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]

        # Get vertical position of pupil in left eye relative to the top and bottom borders
        left_eye_landmarks = [one_face_landmark_points[idx] for idx in left_eye_indices]
        vertical_position_left_eye = calculate_vertical_position(left_eye_landmarks, window_h)

        # Perform scrolling based on the vertical position of the left eye's pupil
        if vertical_position_left_eye < 0.4:
            print("Looking Up - Smooth Scroll Up")
            pyautogui.scroll(scroll_step)  # Scroll up in larger steps
            time.sleep(scroll_delay)  # Short delay for smooth scrolling
        elif vertical_position_left_eye > 0.6:
            print("Looking Down - Smooth Scroll Down")
            pyautogui.scroll(-scroll_step)  # Scroll down in larger steps
            time.sleep(scroll_delay)  # Short delay for smooth scrolling

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
