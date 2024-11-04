import cv2
import mediapipe as mp
import time
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize webcam
cam = cv2.VideoCapture(0)

# Eye landmark indices for MediaPipe Face Mesh
left_eye_indices = [33, 160, 158, 133, 153, 144]  # Landmarks for left eye
right_eye_indices = [362, 385, 387, 263, 373, 380]  # Landmarks for right eye

# Blink detection delay parameters
blink_detected = False
last_blink_time = 0
blink_delay = 1  # Delay in seconds before next blink can be detected

# Function to calculate the distance between two points
def calculate_distance(point1, point2, window_w, window_h):
    dx = (point1.x - point2.x) * window_w
    dy = (point1.y - point2.y) * window_h
    return math.sqrt(dx**2 + dy**2)

# Function to determine if eyes are closed with dynamic threshold
def are_eyes_closed(landmarks, window_w, window_h):
    # Calculate the height of the eyes
    left_eye_top = landmarks[left_eye_indices[1]].y
    left_eye_bottom = landmarks[left_eye_indices[4]].y
    right_eye_top = landmarks[right_eye_indices[1]].y
    right_eye_bottom = landmarks[right_eye_indices[4]].y

    # Calculate eye heights
    left_eye_height = left_eye_bottom - left_eye_top
    right_eye_height = right_eye_bottom - right_eye_top

    # Calculate face distance using eye corners (landmarks 33 and 362)
    eye_corner_distance = calculate_distance(
        landmarks[33], landmarks[362], window_w, window_h
    )

    # Adjust blink threshold based on distance
    base_blink_threshold = 0.029  # Initial threshold for blink detection
    adjusted_blink_threshold = base_blink_threshold * (50 / eye_corner_distance)

    # Return True if both eyes are closed based on adjusted threshold
    return left_eye_height < adjusted_blink_threshold and right_eye_height < adjusted_blink_threshold

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

        # Get the current time
        current_time = time.time()

        # Check if both eyes are closed and if enough time has passed since the last blink
        if are_eyes_closed(one_face_landmark_points, window_w, window_h) and (current_time - last_blink_time > blink_delay):
            print("Both eyes are closed - Detected Blink!")
            last_blink_time = current_time  # Update the last blink time
            blink_detected = True
        else:
            blink_detected = False
            print("Eyes are open")

        # Draw circles on the eye landmarks
        for idx in left_eye_indices:
            x = int(one_face_landmark_points[idx].x * window_w)
            y = int(one_face_landmark_points[idx].y * window_h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        for idx in right_eye_indices:
            x = int(one_face_landmark_points[idx].x * window_w)
            y = int(one_face_landmark_points[idx].y * window_h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # Show the video stream with eye landmarks
    cv2.imshow("Eye Blink Detection with Dynamic Threshold", image)

    # Break the loop on 'ESC' key press
    key = cv2.waitKey(1)
    if key == 27:  # Escape key
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
