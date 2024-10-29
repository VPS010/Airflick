import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize webcam
cam = cv2.VideoCapture(0)

# Eye landmark indices for MediaPipe Face Mesh
left_eye_indices = [33, 160, 158, 133, 153, 144]  # Landmarks for left eye
right_eye_indices = [362, 385, 387, 263, 373, 380]  # Landmarks for right eye
outer_left_eye_index = 33  # Outer corner of the left eye
outer_right_eye_index = 263  # Outer corner of the right eye

# Base blink threshold
base_blink_threshold = 0.003

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

# Function to determine if eyes are closed with adjusted blink threshold
def are_eyes_closed(landmarks, blink_threshold):
    # Calculate the height of the eyes
    left_eye_top = landmarks[left_eye_indices[1]].y
    left_eye_bottom = landmarks[left_eye_indices[4]].y
    right_eye_top = landmarks[right_eye_indices[1]].y
    right_eye_bottom = landmarks[right_eye_indices[4]].y

    # Calculate eye heights
    left_eye_height = left_eye_bottom - left_eye_top
    right_eye_height = right_eye_bottom - right_eye_top

    # Return True if both eyes are closed
    return left_eye_height < blink_threshold and right_eye_height < blink_threshold

while True:
    ret, image = cam.read()

    if not ret:
        break

    # Convert BGR image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect facial landmarks
    processed_image = face_mesh.process(rgb_image)
    all_faces_landmark_points = processed_image.multi_face_landmarks

    if all_faces_landmark_points:
        one_face_landmark_points = all_faces_landmark_points[0].landmark
        
        # Calculate the distance between outer corners of eyes as an indicator of face distance
        outer_left_eye = one_face_landmark_points[outer_left_eye_index]
        outer_right_eye = one_face_landmark_points[outer_right_eye_index]
        eye_distance = calculate_distance(outer_left_eye, outer_right_eye)

        # Adjust blink threshold based on eye distance
        # If the face is farther, increase the threshold, else reduce it
        adjusted_blink_threshold = base_blink_threshold * (1 / (1.05*eye_distance))

        # Check if both eyes are closed with the adjusted threshold
        if are_eyes_closed(one_face_landmark_points, adjusted_blink_threshold):
            print("Both eyes are closed - Detected Blink!")
        else:
            print("Eyes are open")

        # Draw circles on the eye landmarks
        for idx in left_eye_indices:
            x = int(one_face_landmark_points[idx].x * image.shape[1])
            y = int(one_face_landmark_points[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        for idx in right_eye_indices:
            x = int(one_face_landmark_points[idx].x * image.shape[1])
            y = int(one_face_landmark_points[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # Show the video stream with eye landmarks
    cv2.imshow("Eye Blink Detection with Distance Adjustment", image)

    # Break the loop on 'ESC' key press
    key = cv2.waitKey(1)
    if key == 27:  # Escape key
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
