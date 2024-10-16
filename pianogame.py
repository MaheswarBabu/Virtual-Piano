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
    pygame.mixer.Sound('C.wav'),   # C
    pygame.mixer.Sound('D.wav'),   # D
    pygame.mixer.Sound('E.wav'),   # E
    pygame.mixer.Sound('F.wav'),   # F
    pygame.mixer.Sound('G.wav'),   # G
    pygame.mixer.Sound('A.wav'),   # A
    pygame.mixer.Sound('B.wav'),   # B
    pygame.mixer.Sound('C2.wav'),  # C2
    pygame.mixer.Sound('D2.wav'),  # D2
    pygame.mixer.Sound('E2.wav')   # E2
]

black_sounds = [
    pygame.mixer.Sound('C#.wav'),  # C#
    pygame.mixer.Sound('D#.wav'),  # D#
    pygame.mixer.Sound('F#.wav'),  # F#
    pygame.mixer.Sound('G#.wav'),  # G#
    pygame.mixer.Sound('A#.wav'),  # A#
    pygame.mixer.Sound('C#2.wav'), # C#2
    pygame.mixer.Sound('D#2.wav')  # D#2
]

# Open the webcam and set the resolution to 1920x1080
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Finger tips landmarks indices for all fingers (thumb, index, middle, ring, pinky)
fingertips_indices = [4, 8, 12, 16, 20]

# Store the state of pressed keys
num_white_keys = 10  # Adjusted to match the second code
num_black_keys = 7
key_pressed_state = [False] * (num_white_keys + num_black_keys)

# Store the key being pressed by each finger
finger_on_key = [-1] * len(fingertips_indices)

# Minimum time to allow repeated sound and key visual state
key_delay = 5  # 5 second delay between re-pressing the same key
visual_press_duration = 0.70  # Key remains visually pressed for 0.7 second

# Track which key was played last by each finger and when
last_played_time = [0] * (num_white_keys + num_black_keys)
last_visual_press_time = [0] * (num_white_keys + num_black_keys)  # Track visual press state timing

def check_key_press(x, y, white_keys, black_keys, white_key_height, black_key_height):
    # Increased detection range for white keys
    for i, (x_start, x_end) in enumerate(white_keys):
        if x_start < x < x_end and y < white_key_height + 50:
            return i
    # Increased detection range for black keys
    for i, (x_start, x_end) in enumerate(black_keys):
        if x_start < x < x_end and y < black_key_height + 50:
            return i + len(white_keys)
    return None

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Define key sizes and positions based on the image layout
    white_key_width = w // 10  # 10 white keys
    white_key_height = h // 3
    black_key_width = int(white_key_width * 0.6)  # Black keys are about 60% the width of white keys
    black_key_height = int(white_key_height * 0.6)

    # Define white key positions
    white_keys = [(i * white_key_width, (i + 1) * white_key_width) for i in range(num_white_keys)]

    # Define black key positions
    black_keys = [
        (int((i + 0.75) * white_key_width), int((i + 1.25) * white_key_width))
        for i in [0, 1, 3, 4, 5, 7, 8]  # Positions of black keys in an octave
    ]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw white piano keys
    for i, (x_start, x_end) in enumerate(white_keys):
        current_time = time.time()
        color = (255, 255, 255)  # Default white color
        if key_pressed_state[i] and (current_time - last_visual_press_time[i] < visual_press_duration):
            color = (192, 192, 192)  # Change to grey when visually pressed
        cv2.rectangle(frame, (x_start, 0), (x_end, white_key_height), color, -1)
        cv2.rectangle(frame, (x_start, 0), (x_end, white_key_height), (0, 0, 0), 2)  # Borders

    # Draw black piano keys
    for i, (x_start, x_end) in enumerate(black_keys):
        current_time = time.time()
        color = (0, 0, 0)  # Default black color
        if key_pressed_state[i + num_white_keys] and (current_time - last_visual_press_time[i + num_white_keys] < visual_press_duration):
            color = (128, 128, 128)  # Change to grey when visually pressed
        cv2.rectangle(frame, (x_start, 0), (x_end, black_key_height), color, -1)
        cv2.rectangle(frame, (x_start, 0), (x_end, black_key_height), (0, 0, 0), 2)

    # Check if hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for i, tip_index in enumerate(fingertips_indices):
                x_tip = int(hand_landmarks.landmark[tip_index].x * w)
                y_tip = int(hand_landmarks.landmark[tip_index].y * h)

                key_index = check_key_press(x_tip, y_tip, white_keys, black_keys, white_key_height, black_key_height)
                current_time = time.time()

                # If finger is on a key and the timer has expired or first time pressed
                if key_index is not None and (finger_on_key[i] != key_index or current_time - last_played_time[key_index] > key_delay):
                    last_played_time[key_index] = current_time
                    finger_on_key[i] = key_index  # Update finger's current key

                    # Play sound for white or black key
                    if key_index < num_white_keys:
                        white_sounds[key_index].play()
                    else:
                        black_sounds[key_index - num_white_keys].play()

                    # Mark the key as pressed and track visual press time
                    key_pressed_state[key_index] = True
                    last_visual_press_time[key_index] = current_time

                # If finger leaves the key, reset state for that key
                if key_index is None:
                    if finger_on_key[i] != -1:
                        finger_on_key[i] = -1

    else:
        # Reset key states if no hands are detected
        finger_on_key = [-1] * len(fingertips_indices)

    cv2.imshow('Virtual Piano', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.quit()
