import sys
import mediapipe as mp
import cv2 as cv
import keyboard as kb

mp_mesh = mp.solutions.hands.Hands()
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

action_active = False
X_THRES_LEFT = int(1920/3)
X_THRES_RIGHT  = 2*int(1920/3)
cam = cv.VideoCapture(0)

while True:
    _, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    res = mp_mesh.process(img)
    # for x in res.multi_face_landmarks:
    if res.multi_hand_landmarks:
        for i in res.multi_hand_landmarks:
            # print(len(i.landmark))
            mp_draw.draw_landmarks(
                img,
                i,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.DrawingSpec(
                    color=(255, 255, 255),
                    thickness=5,
                    circle_radius=7
                ),
                mp_drawing_styles.DrawingSpec(
                    color=(0, 0, 0),
                    thickness=2,
                    circle_radius=0
                )
            )

            mid_fin_x = i.landmark[12].x * 1920

            if X_THRES_LEFT < mid_fin_x < X_THRES_RIGHT:
                action_active = True
            elif action_active and mid_fin_x < X_THRES_LEFT:
                print("Switching left")
                kb.press_and_release('alt+shift+n')
                action_active = False
            elif action_active and mid_fin_x > X_THRES_RIGHT:
                print("switching right")
                kb.press_and_release('alt+shift+m')
                action_active = False

    cv.namedWindow("img", cv.WINDOW_NORMAL)
    cv.resizeWindow("img", 1366, 768)
    cv.imshow("img", img)
    k = cv.waitKey(10)
    if k == 27:
        break

cam.release()