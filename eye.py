import cv2
import mediapipe as mp
import pyautogui

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cam = cv2.VideoCapture(0)

scroll_speed = 10

def calculate_vertical_position(eye_landmarks, window_h):
    top_of_eye = int(eye_landmarks[1].y * window_h)  
    bottom_of_eye = int(eye_landmarks[4].y * window_h)  
    pupil_y = int(eye_landmarks[0].y * window_h)

    relative_position = (pupil_y - top_of_eye) / (bottom_of_eye - top_of_eye)
    return relative_position

while True:
    ret, image = cam.read()

    if not ret:
        break

    window_h, window_w, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    processed_image = face_mesh.process(rgb_image)
    all_faces_landmark_points = processed_image.multi_face_landmarks

    if all_faces_landmark_points:
        one_face_landmark_points = all_faces_landmark_points[0].landmark

        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]

        left_eye_landmarks = [one_face_landmark_points[idx] for idx in left_eye_indices]
        vertical_position_left_eye = calculate_vertical_position(left_eye_landmarks, window_h)

        if vertical_position_left_eye < 0.4:
            print("Looking Up - Scroll Up")
            pyautogui.scroll(scroll_speed) 
        elif vertical_position_left_eye > 0.6:
            print("Looking Down - Scroll Down")
            pyautogui.scroll(-scroll_speed) 

        for idx in left_eye_indices:
            x = int(one_face_landmark_points[idx].x * window_w)
            y = int(one_face_landmark_points[idx].y * window_h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        for idx in right_eye_indices:
            x = int(one_face_landmark_points[idx].x * window_w)
            y = int(one_face_landmark_points[idx].y * window_h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Eye Detection", image)

    key = cv2.waitKey(1)
    if key == 27: 
        break

cam.release()
cv2.destroyAllWindows()
