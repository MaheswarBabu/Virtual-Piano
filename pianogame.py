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

# Open the webcam and set the resolution to 1920x1080
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Finger tips landmarks indices for all fingers (thumb, index, middle, ring, pinky)
fingertips_indices = [4, 8, 12, 16, 20]

# Store the state of pressed keys
key_pressed_state = [False] * (14 + 5 + 14 + 5)  # 14 white + 5 black in each octave

# Store the time when each key was last pressed
key_press_timestamps = [0] * len(key_pressed_state)

# Store the key being pressed by each finger
finger_on_key = [-1] * len(fingertips_indices)

# Minimum time to allow repeated sound
key_delay = 5.0  # 5 seconds

def check_key_press(x, y, white_keys, black_keys, white_key_height, black_key_height):
    # Check white keys
    for i, (x_start, x_end) in enumerate(white_keys):
        if x_start < x < x_end and y < white_key_height:  # White key area
            return i
    # Check black keys
    for i, (x_start, x_end) in enumerate(black_keys):
        if x_start < x < x_end and y < black_key_height:
            return i + len(white_keys)
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape  # Get the webcam frame dimensions

    # Dynamically calculate key sizes based on the frame width
    num_white_keys = 14  # Two octaves (7 keys per octave)
    white_key_width = w // num_white_keys  # Dynamically adjust white key width
    white_key_height = h // 3  # Set white key height to 1/3 of frame height
    black_key_height = int(white_key_height * 0.65)  # Black key height is 65% of white key height
    black_key_width = white_key_width // 2  # Black key is half the width of white keys

    # Define white and black key ranges for two octaves
    white_keys = [(i * white_key_width, (i + 1) * white_key_width) for i in range(num_white_keys)]
    black_keys = [
        (int(0.75 * white_key_width), int(1.25 * white_key_width)),  # C#
        (int(1.75 * white_key_width), int(2.25 * white_key_width)),  # D#
        (int(3.75 * white_key_width), int(4.25 * white_key_width)),  # F#
        (int(4.75 * white_key_width), int(5.25 * white_key_width)),  # G#
        (int(5.75 * white_key_width), int(6.25 * white_key_width)),  # A#
        (int(7.75 * white_key_width), int(8.25 * white_key_width)),  # C#2
        (int(8.75 * white_key_width), int(9.25 * white_key_width)),  # D#2
        (int(10.75 * white_key_width), int(11.25 * white_key_width)),  # F#2
        (int(11.75 * white_key_width), int(12.25 * white_key_width)),  # G#2
        (int(12.75 * white_key_width), int(13.25 * white_key_width))   # A#2
    ]

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    # Draw white piano keys (on the top of the frame)
    for i, (x_start, x_end) in enumerate(white_keys):
        color = (255, 255, 255)  # Default color for white keys
        if key_pressed_state[i]:
            color = (192, 192, 192)  # Grey when pressed
        cv2.rectangle(frame, (x_start, 0), (x_end, white_key_height), color, -1)  # Keys at the top
        cv2.rectangle(frame, (x_start, 0), (x_end, white_key_height), (0, 0, 0), 2)  # Key border

    # Draw black piano keys (on top of the white keys)
    for i, (x_start, x_end) in enumerate(black_keys):
        color = (0, 0, 0)  # Default color for black keys
        if key_pressed_state[i + len(white_keys)]:
            color = (128, 128, 128)  # Grey when pressed
        cv2.rectangle(frame, (x_start, 0), (x_end, black_key_height), color, -1)  # Keys at the top

    # Check if hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check fingertip positions for each finger
            for i, tip_index in enumerate(fingertips_indices):
                x_tip = int(hand_landmarks.landmark[tip_index].x * w)
                y_tip = int(hand_landmarks.landmark[tip_index].y * h)

                # Check if a key is pressed
                key_index = check_key_press(x_tip, y_tip, white_keys, black_keys, white_key_height, black_key_height)
                if key_index is not None:
                    current_time = time.time()

                    if finger_on_key[i] != key_index:  # New key detected, play sound and start timer
                        finger_on_key[i] = key_index
                        key_press_timestamps[key_index] = current_time
                        
                        # Play the corresponding sound
                        if key_index < len(white_sounds):
                            white_sounds[key_index].play()
                        elif key_index < len(white_sounds) + len(black_sounds):
                            black_sounds[key_index - len(white_sounds)].play()

                        key_pressed_state[key_index] = True

                    elif current_time - key_press_timestamps[key_index] > key_delay:
                        # After the delay, just update the timestamp to prevent sound replay
                        key_press_timestamps[key_index] = current_time
                else:
                    finger_on_key[i] = -1  # Finger is not on any key
                    key_pressed_state = [False] * len(key_pressed_state)  # Reset the key pressed states

    # Display the frame with piano keys and hand detection
    cv2.imshow('Virtual Piano', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.quit()
