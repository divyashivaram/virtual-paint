import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

# frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

frameWidth = 1280
frameHeight = 720

cap.set(3, 1280)
cap.set(4, 720)

xp, yp = 0, 0  # initial positions of the pointer

# Canvas to draw
imgCanvas = np.zeros((frameHeight, frameWidth, 3), np.uint8)
imgCanvas.fill(255)


with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1)

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)

        # Draw the hand annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Detects hand(s) dexterity in view
                hand = [results.multi_handedness[i].classification[0].label for i in range(
                    len(results.multi_handedness))]

                normalizedLandmark = results.multi_hand_landmarks[0].landmark[
                    mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                pixelCoordinatesLandmark = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                    normalizedLandmark.x, normalizedLandmark.y, frameWidth, frameHeight)

                print('pixelCoordinatesLandmark', pixelCoordinatesLandmark)
                if pixelCoordinatesLandmark:
                    x1, y1 = pixelCoordinatesLandmark
                else:
                    x1, y1 = 0, 0
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

                # When finger appears on screen, a line os drawn from 0,0
                # To fix, initial values are set to current coordinates
                # TODO: However this should happen when paint mode changes and returns too. Reset while defining different modes
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                cv2.line(img, (xp, yp), (x1, y1), (0, 0, 0), 10)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), (0, 0, 0), 10)

                xp, yp = x1, y1

                # Draw landmarks on video stream
                # mp_drawing.draw_landmarks(
                #     img,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style())

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', img)
        cv2.imshow('Canvas', imgCanvas)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()


def is_only_index_finger_up(landmarks):
    pass
