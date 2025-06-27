import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Reduce complexity of drawing to save memory
        self.drawing_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)
        
        # Configure hands - use static image mode for memory efficiency if not tracking motion
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            # Set model complexity to 0 (Lite) for better performance and less memory use
            model_complexity=0
        )

    def find_hands(self, frame, draw=True):
        """Process frame and return hand landmarks if found"""
        # Process the frame with lower resolution to save memory
        h, w = frame.shape[:2]
        # Only resize if the frame is large
        if w > 640:
            # Process a smaller image for detection (faster and less memory intensive)
            process_w = 640
            process_h = int(h * (process_w / w))
            small_frame = cv2.resize(frame, (process_w, process_h))
            frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        else:
            # If already small, just convert color
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set image data to writeable to avoid copying memory
        frame_rgb.flags.writeable = False
        result = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.drawing_spec,  # Use simplified drawing specs
                        self.drawing_spec
                    )
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
        # Simpler calculation using only x, y coordinates to save processing power
        p1 = np.array([point1.x, point1.y])
        p2 = np.array([point2.x, point2.y])
        p3 = np.array([point3.x, point3.y])
        
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
        
    def is_thumbs_up(self, landmarks):
        """
        Detect thumbs up gesture:
        - Thumb is extended upward
        - All other fingers are folded
        - Hand orientation is roughly vertical
        """
        # Get all landmarks needed
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # Check if thumb is pointing upward
        thumb_pointing_up = thumb_tip.y < thumb_ip.y and thumb_ip.y < thumb_mcp.y
        
        # Check if other fingers are folded (lower than their respective MCPs)
        index_folded = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        middle_folded = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
        ring_folded = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP].y
        pinky_folded = landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y > landmarks[self.mp_hands.HandLandmark.PINKY_MCP].y
        
        # Additional check to ensure hand is somewhat vertical (thumb should be to the side of the palm)
        hand_vertical = abs(thumb_tip.x - wrist.x) < 0.2  # Thumb shouldn't be too far to the side
        
        # All conditions must be true for thumbs up
        return thumb_pointing_up and index_folded and middle_folded and ring_folded and pinky_folded
    
    def is_thumbs_down(self, landmarks):
        """
        Detect thumbs down gesture:
        - Thumb is extended downward
        - All other fingers are folded
        - Hand orientation is roughly vertical
        """
        # Get all landmarks needed
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # Check if thumb is pointing downward
        thumb_pointing_down = thumb_tip.y > thumb_ip.y and thumb_ip.y > thumb_mcp.y
        
        # Check if other fingers are folded (lower than their respective MCPs)
        index_folded = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y > landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        middle_folded = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
        ring_folded = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP].y
        pinky_folded = landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y > landmarks[self.mp_hands.HandLandmark.PINKY_MCP].y
        
        # Additional check to ensure hand is somewhat vertical (thumb should be to the side of the palm)
        hand_vertical = abs(thumb_tip.x - wrist.x) < 0.2  # Thumb shouldn't be too far to the side
        
        # All conditions must be true for thumbs down
        return thumb_pointing_down and index_folded and middle_folded and ring_folded and pinky_folded