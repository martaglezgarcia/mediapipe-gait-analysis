
# implementing two cameras

# error:   the video is lagged

# this code uses a Canon 1200D and the computer camera as webcams

import mediapipe as mp 
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# landmarks
keypoints = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
frame_shape = [720, 1280]

# use canon as webcam
v1 = cv2.VideoCapture(2)
# use computer webcam
v2 = cv2.VideoCapture(0)

# get height and width
def get_hw(video):
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height

w1,h1 = get_hw(v1)
w2,h2 = get_hw(v2)

## MEDIAPIPE POSE
with mp_pose.Pose(static_image_mode=False) as pose:
    while True:
        ret1, frame1 = v1.read()
        ret2, frame2 = v2.read()
        

        ## Convert videos to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame1.flags.writeable = False 
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame2.flags.writeable = False 

        ## Make detection
        results1 = pose.process(frame1)
        results2 = pose.process(frame2)

        ## Recolor
        frame1.flags.writeable = True
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
        frame2.flags.writeable = True
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

        ## Get landmarks


        ## Render detections
        mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

        mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
        

        ## Show videos
        cv2.namedWindow('window 1', cv2.WINDOW_NORMAL & cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('window 1', w1, h1)
        cv2.imshow('window 1', frame1)

        cv2.namedWindow('window 2', cv2.WINDOW_NORMAL & cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('window 2', w2, h2)
        cv2.imshow('window 2', frame2)


        if cv2.waitKey(10) & 0xFF == ord('q'):  #pressing q will close the windows
            break


