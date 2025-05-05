import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def find_hands(self, frame, draw=True):
        """Process frame and return hand landmarks if found"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            return frame, result.multi_hand_landmarks
        return frame, None
    
    def get_finger_midpoint(self, landmarks, index, middle):
        """Get midpoint between two finger landmarks"""
        x = (landmarks[index].x + landmarks[middle].x) / 2
        y = (landmarks[index].y + landmarks[middle].y) / 2
        return x, y

    def is_finger_raised(self, landmarks, tip_idx, pip_idx):
        """Check if a finger is raised by comparing tip height with PIP joint"""
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    def get_angle(self, a, b, c):
        """Calculate angle between three points (landmarks)"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def get_distance(self, points):
        """Calculate distance between two points (landmarks)"""
        point1, point2 = points
        x1, y1 = point1.x, point1.y
        x2, y2 = point2.x, point2.y
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 100  # Scale for better thresholding

    def is_finger_folded(self, landmarks, tip_idx, pip_idx, mcp_idx):
        """
        More robust method to check if a finger is folded
        Uses the angle between joints to determine finger state
        """
        # Get coordinates of the three joints
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]
        
        # Calculate vectors between joints
        v1 = np.array([tip.x - pip.x, tip.y - pip.y, tip.z - pip.z])
        v2 = np.array([mcp.x - pip.x, mcp.y - pip.y, mcp.z - pip.z])
        
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Calculate dot product and angle
        dot_product = np.dot(v1, v2)
        # Clamp to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product) * 180.0 / np.pi
        
        # For thumb, use a different threshold
        if tip_idx == self.mp_hands.HandLandmark.THUMB_TIP.value:
            return angle < 90  # Thumb is folded if angle is small
        
        # For other fingers, folded when angle is large
        return angle > 90
    
    def is_index_finger_straight(self, landmarks):
        """
        Check if the index finger is straight by calculating angles between all joints
        Returns True if the index finger is straight, False if it's bent
        """
        # Get coordinates of all index finger joints
        tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        dip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_DIP]
        pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        # Calculate angle between tip-dip-pip 
        angle1 = self.calculate_angle_between_points(tip, dip, pip)
        
        # Calculate angle between dip-pip-mcp
        angle2 = self.calculate_angle_between_points(dip, pip, mcp)
        
        # The finger is straight if both angles are extremely close to 180 degrees
        # Using 175 degrees as a threshold for "straightness" - extremely strict
        # This will detect even the slightest bends in the finger
        return angle1 > 175 and angle2 > 175
    
    def calculate_angle_between_points(self, point1, point2, point3):
        """
        Calculate the angle in degrees between three points
        The angle is calculated at point2
        """
        # Convert landmarks to numpy arrays
        p1 = np.array([point1.x, point1.y, point1.z])
        p2 = np.array([point2.x, point2.y, point2.z])
        p3 = np.array([point3.x, point3.y, point3.z])
        
        # Calculate two vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Calculate dot product
        dot_product = np.dot(v1, v2)
        
        # Clamp to avoid numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate angle in degrees
        angle = np.arccos(dot_product) * 180.0 / np.pi
        
        return angle
        
    def is_index_finger_only(self, landmarks):
        """
        Check if only the index finger is extended and others are folded
        Using the more robust angle-based method
        """
        # ... existing code ...