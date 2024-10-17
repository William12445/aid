import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize Mediapipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Track both hands
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
right_hand_fist_detected = False  # Track if right hand fist has been detected
cursor_speed = 2 # Adjust this value to control the cursor speed

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

# Function to get the center of the palm
def get_palm_center(landmarks):
    # Use the center of the palm (between wrist and thumb base) as the palm center
    thumb_base = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    return (thumb_base.x + wrist.x) / 2, (thumb_base.y + wrist.y) / 2

# Function to move the cursor with boundaries
def move_cursor_with_boundaries(dx, dy):
    global prev_right_x, prev_right_y
    # Calculate the new position
    new_x = int(prev_right_x + dx)
    new_y = int(prev_right_y + dy)

    # Restrict movement within screen bounds
    new_x = max(0, min(screen_width - 1, new_x))
    new_y = max(0, min(screen_height - 1, new_y))

    # Move the cursor
    pyautogui.moveTo(new_x, new_y)
    
    # Update previous position
    prev_right_x = new_x
    prev_right_y = new_y

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for natural movement
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    # Get image dimensions
    frame_height, frame_width, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = hand_info.classification[0].label  # 'Left' or 'Right'

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
                        pyautogui.scroll(50)
                    elif wrist_y > prev_left_y:  # Scroll down
                        pyautogui.scroll(-50)

                prev_left_y = wrist_y

            elif hand_label == 'Right':  # Right hand controls cursor and clicks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                # Get the center of the palm to control cursor movement
                palm_center_x, palm_center_y = get_palm_center(hand_landmarks.landmark)
                palm_x = int(palm_center_x * screen_width)
                palm_y = int(palm_center_y * screen_height)

                # Scale cursor movement based on speed factor
                if prev_right_x is not None and prev_right_y is not None:
                    dx = (palm_x - prev_right_x) * cursor_speed
                    dy = (palm_y - prev_right_y) * cursor_speed
                    print(f"Cursor Movement: dx={dx}, dy={dy}")  # Debug cursor movement
                    move_cursor_with_boundaries(dx, dy)  # Move cursor with boundaries

                # Check for fist to perform a click
                if is_fist(hand_landmarks.landmark):
                    if not right_hand_fist_detected:
                        pyautogui.click()  # Perform a click
                        right_hand_fist_detected = True  # Mark fist as detected
                else:
                    right_hand_fist_detected = False  # Reset when hand returns to normal

    # Display the frame with landmarks
    cv2.imshow('Hand Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
