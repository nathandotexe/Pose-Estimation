import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to detect pose
def detect_pose(landmarks):
    detected_pose = None
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate vertical midpoint of shoulders
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

    if nose.y < shoulder_y - 0.05:  # Looking Up
        detected_pose = "Looking Up"
    elif nose.y > shoulder_y + 0.05:  # Looking Down
        detected_pose = "Looking Down"
    elif nose.x < left_shoulder.x - 0.1:  # Turning Left
        detected_pose = "Turning Left"
    elif nose.x > right_shoulder.x + 0.1:  # Turning Right
        detected_pose = "Turning Right"
    
    return detected_pose

# Function to count fingers and detect gestures
def detect_hand_gestures(hand_landmarks, image_width, image_height):
    # Tip and base landmark indices
    finger_tips = [4, 8, 12, 16, 20]
    finger_bases = [2, 6, 10, 14, 18]

    # Extract coordinates
    landmarks = [(lm.x * image_width, lm.y * image_height) for lm in hand_landmarks]

    raised_fingers = 0
    for tip, base in zip(finger_tips[1:], finger_bases[1:]): 
        if landmarks[tip][1] < landmarks[base][1]:  
            raised_fingers += 1

    thumb_is_up = landmarks[4][1] < landmarks[2][1]

    gesture = None
    if raised_fingers == 0 and thumb_is_up:
        gesture = "Thumbs Up"
    elif raised_fingers == 0 and not thumb_is_up:
        gesture = "Thumbs Down"
    elif raised_fingers == 5:
        gesture = "Open Hand"
    
    return raised_fingers, thumb_is_up, gesture

def detect_heart_gesture(hand_landmarks_left, hand_landmarks_right, image_width, image_height):
  
    left_index_tip = hand_landmarks_left[8]
    right_index_tip = hand_landmarks_right[8]

    left_tip = (left_index_tip.x * image_width, left_index_tip.y * image_height)
    right_tip = (right_index_tip.x * image_width, right_index_tip.y * image_height)

    distance = np.linalg.norm(np.array(left_tip) - np.array(right_tip))
    if distance < 50:
        return "Heart Gesture"
    return None

def count_fingers(hand_landmarks, width, height):
    finger_tips = [4, 8, 12, 16, 20]  
    finger_bases = [2, 6, 10, 14, 18]

    coords = [(lm.x * width, lm.y * height) for lm in hand_landmarks]
    fingers_up = 0

    for tip, base in zip(finger_tips[1:], finger_bases[1:]):  # Skip thumb for now
        if coords[tip][1] < coords[base][1]:  # Tip is above base
            fingers_up += 1

    thumb_tip, thumb_base = coords[4], coords[2]
    thumb_direction = (thumb_tip[0] - thumb_base[0]) > 0  # Left vs. right hand
    if thumb_direction:
        fingers_up += 1

    return fingers_up

cap = cv2.VideoCapture(0) 
pose = mp_pose.Pose()
hands = mp_hands.Hands()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(rgb_frame)
    hands_result = hands.process(rgb_frame)
    
    annotated_frame = frame.copy()

    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        landmarks = pose_result.pose_landmarks.landmark
        pose_detected = detect_pose(landmarks)
        if pose_detected:
            cv2.putText(annotated_frame, f"Pose: {pose_detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if hands_result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
            mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers_count = count_fingers(hand_landmarks.landmark, width, height)
            handedness = hands_result.multi_handedness[idx].classification[0].label
            hand_label = "Right Hand" if handedness == "Right" else "Left Hand"

            coords = (int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width),
                      int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height))
            cv2.putText(
                annotated_frame,
                f"{hand_label}: {fingers_count} fingers",
                (coords[0] - 50, coords[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            raised_fingers, thumb_is_up, gesture = detect_hand_gestures(hand_landmarks.landmark, width, height)
            if gesture:
                cv2.putText(
                    annotated_frame,
                    f"Gesture: {gesture}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )

        if len(hands_result.multi_hand_landmarks) == 2:
            left_hand = hands_result.multi_hand_landmarks[0]
            right_hand = hands_result.multi_hand_landmarks[1]
            heart_gesture = detect_heart_gesture(left_hand.landmark, right_hand.landmark, width, height)
            if heart_gesture:
                cv2.putText(
                    annotated_frame,
                    f"Gesture: {heart_gesture}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

    combined_view = np.hstack((frame, annotated_frame))
    cv2.imshow("Camera & Enhanced Gestures", combined_view)
    
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
