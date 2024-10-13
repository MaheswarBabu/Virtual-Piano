import cv2
import mediapipe as mp
import pygame
import time

# Initialize Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pygame for sound
pygame.init()
pygame.mixer.init()

# Load piano sounds for white and black keys
white_sounds = [
    pygame.mixer.Sound('C.wav'),    # C
    pygame.mixer.Sound('D.wav'),    # D
    pygame.mixer.Sound('E.wav'),    # E
    pygame.mixer.Sound('F.wav'),    # F
    pygame.mixer.Sound('G.wav'),    # G
    pygame.mixer.Sound('A.wav'),    # A
    pygame.mixer.Sound('B.wav')     # B
]

black_sounds = [
    pygame.mixer.Sound('C#.wav'),   # C#
    pygame.mixer.Sound('D#.wav'),   # D#
    pygame.mixer.Sound('F#.wav'),   # F#
    pygame.mixer.Sound('G#.wav'),   # G#
    pygame.mixer.Sound('A#.wav')    # A#
]

# Open the webcam
cap = cv2.VideoCapture(0)

# Define screen width and height
screen_width = 1920
screen_height = 1080

# White keys settings
white_key_height = 200
white_key_width = 60  # Increased width of white keys

white_keys = [
    (0, 60),    # C
    (60, 120),   # D
    (120, 180),  # E
    (180, 240), # F
    (240, 300), # G
    (300, 360), # A
    (360, 420)  # B
]

white_key_labels = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

# Black keys settings (Visual remains the same, detection area will be larger)
black_key_height = 130  # Visual height of black keys remains the same
black_key_width = 25    # Visual width of black keys remains the same

# Adjust the placement of black keys for better detection
black_keys_visual = [
    (45, 70),   # C# (Visual bounds)
    (105, 130), # D#
    (225, 250), # F#
    (285, 310), # G#
    (345, 370)  # A#
]

black_key_labels = ['C#', 'D#', 'F#', 'G#', 'A#']

# Detection area for black keys (Larger than visual bounds)
black_keys_detection = [
    (40, 75),   # C# (Wider detection area)
    (100, 135), # D#
    (220, 255), # F#
    (280, 315), # G#
    (340, 375)  # A#
]

# Finger tips landmarks indices for all fingers (thumb, index, middle, ring, pinky)
fingertips_indices = [4, 8, 12, 16, 20]

# Store the state of pressed keys
key_pressed_state = [False] * (len(white_keys) + len(black_keys_visual))

# Store the time when each key was last pressed
key_press_timestamps = [0] * (len(white_keys) + len(black_keys_visual))

# Minimum time in seconds to wait before allowing a key to be pressed again
key_delay = 0.5  # 0.5 seconds

def check_key_press(x, y):
    # Check white keys
    for i, (x_start, x_end) in enumerate(white_keys):
        if x_start < x < x_end and y < white_key_height:  # White key area
            return i
    # Check black keys using the larger detection area
    for i, (x_start, x_end) in enumerate(black_keys_detection):
        if x_start < x < x_end and y < black_key_height + 20:  # Larger detection area for black keys
            return i + len(white_keys)  # Offset by the number of white keys
    return None

# Set the size of the window to match the new dimensions
cv2.namedWindow('Virtual Piano', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Virtual Piano', screen_width, screen_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    # Draw white piano keys
    for i, (x_start, x_end) in enumerate(white_keys):
        color = (255, 255, 255)  # Default color for white keys
        if key_pressed_state[i]:
            color = (192, 192, 192)  # Grey when pressed
        cv2.rectangle(frame, (x_start, 0), (x_end, white_key_height), color, -1)
        cv2.rectangle(frame, (x_start, 0), (x_end, white_key_height), (0, 0, 0), 2)  # Key border
        # Add white key labels
        cv2.putText(frame, white_key_labels[i], (x_start + 10, white_key_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Draw black piano keys (Visual part remains small)
    for i, (x_start, x_end) in enumerate(black_keys_visual):
        color = (0, 0, 0)  # Default color for black keys
        if key_pressed_state[i + len(white_keys)]:
            color = (128, 128, 128)  # Grey when pressed
        cv2.rectangle(frame, (x_start, 0), (x_end, black_key_height), color, -1)
        # Add black key labels
        cv2.putText(frame, black_key_labels[i], (x_start + 5, black_key_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Track which keys are pressed in the current frame
    keys_currently_pressed = [False] * (len(white_keys) + len(black_keys_visual))

    # If hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Iterate over fingertips for all fingers
            for fingertip in fingertips_indices:
                fingertip_landmark = hand_landmarks.landmark[fingertip]
                x = int(fingertip_landmark.x * w)
                y = int(fingertip_landmark.y * h)

                # Check if the finger is over a key region
                key_pressed = check_key_press(x, y)
                if key_pressed is not None:
                    # Get the current time
                    current_time = time.time()

                    # Check if enough time has passed since the last press
                    if not key_pressed_state[key_pressed] and (current_time - key_press_timestamps[key_pressed]) >= key_delay:
                        # Play the sound if the key was not previously pressed and the delay has passed
                        if key_pressed < len(white_keys):
                            white_sounds[key_pressed % len(white_sounds)].play()  # Play white key sound
                        else:
                            black_sounds[(key_pressed - len(white_keys)) % len(black_sounds)].play()  # Play black key sound
                        
                        key_pressed_state[key_pressed] = True
                        key_press_timestamps[key_pressed] = current_time  # Update last press time

                    # Mark this key as pressed in the current frame
                    keys_currently_pressed[key_pressed] = True

                # Draw a circle at the tip of each finger
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

    # Update key press states
    key_pressed_state = keys_currently_pressed

    # Display the frame
    cv2.imshow('Virtual Piano', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()
