import cv2
import mediapipe as mp
import pygame
import time

pygame.mixer.init()
sound_selamat = pygame.mixer.Sound("selamat.mp3")
sound_berjuang = pygame.mixer.Sound("berjuang.mp3")
sound_sukses = pygame.mixer.Sound("sukses.mp3")

last_play = 0
COOLDOWN = 2  

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    fingers.append(hand_landmarks.landmark[tips[0]].x <
                   hand_landmarks.landmark[tips[0] - 1].x)

    for i in range(1, 5):
        fingers.append(
            hand_landmarks.landmark[tips[i]].y <
            hand_landmarks.landmark[tips[i] - 2].y
        )

    return fingers  # [thumb, index, middle, ring, pinky]

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    now = time.time()

    if result.multi_hand_landmarks:
        finger_states = []

        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            finger_states.append(fingers_up(hand))

        if now - last_play > COOLDOWN:

            if len(finger_states) == 1:
                f = finger_states[0]
                if f == [False, True, False, False, True]:
                    sound_selamat.play()
                    last_play = now
                    cv2.putText(frame, "Mode: Selamat", (20,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

                elif f == [False, False, False, False, False]:
                    sound_berjuang.play()
                    last_play = now
                    cv2.putText(frame, "Mode: Berjuang", (20,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

            if len(finger_states) == 2:
                if all(f == [True, False, False, False, False] for f in finger_states):
                    sound_sukses.play()
                    last_play = now
                    cv2.putText(frame, "Mode: Sukses", (20,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    cv2.imshow("Gesture Audio Player", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

