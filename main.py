import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize Mediapipe hand tracking with detection and tracking confidence
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Capture webcam feed
cap = cv2.VideoCapture(0)

# Screen resolution for scaling hand movement to screen size
screen_width, screen_height = pyautogui.size()

# Variables to store previous hand positions and scrolling state
prev_left_y = None
prev_right_x = None
prev_right_y = None
scrolling_enabled = True

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

# Function to check if a fist is made (all fingers except thumb are closed)
def is_fist(landmarks):
    finger_tips_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    finger_base_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP,
    ]
    
    for tip_id, base_id in zip(finger_tips_ids, finger_base_ids):
        if landmarks[tip_id].y < landmarks[base_id].y:
            return False  # If any finger is not closed, it's not a fist
    return True

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Flip the frame for natural movement
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    # Get image dimensions
    frame_height, frame_width, _ = frame.shape

    # Check if any hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = hand_info.classification[0].label  # 'Left' or 'Right'
            print(f"{hand_label} hand detected.")

            if hand_label == 'Left':  # Left hand controls scrolling
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

                # Use the wrist's Y-coordinate to control scrolling
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_y = int(wrist.y * frame_height)

                # Check for fist and scrolling behavior
                if is_fist(hand_landmarks.landmark):
                    scrolling_enabled = False  # Stop scrolling when a fist is detected
                else:
                    scrolling_enabled = True

                if scrolling_enabled and prev_left_y is not None:
                    if wrist_y < prev_left_y:  # Scroll up
                        pyautogui.scroll(10)
                    elif wrist_y > prev_left_y:  # Scroll down
                        pyautogui.scroll(-10)

                prev_left_y = wrist_y

            elif hand_label == 'Right':  # Right hand controls cursor and clicks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                # Use the wrist to control cursor movement
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                wrist_x = int(wrist.x * screen_width)
                wrist_y = int(wrist.y * screen_height)

                # Check for fist to perform a click
                if is_fist(hand_landmarks.landmark):
                    pyautogui.click()  # Perform a click

                # Move cursor
                pyautogui.moveTo(wrist_x, wrist_y)

                prev_right_x = wrist_x
                prev_right_y = wrist_y

    else:
        print("No hands detected.")  # Debugging message

    # Display the frame with landmarks
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
