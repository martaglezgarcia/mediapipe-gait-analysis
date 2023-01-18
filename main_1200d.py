
# this code uses a Canon 1200D as webcam

import mediapipe as mp 
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# use canon as webcam
video = cv2.VideoCapture(1)

# get height and width
def get_hw(video):
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return w, h

width,height = get_hw(video)

## MEDIAPIPE POSE
with mp_pose.Pose(static_image_mode=False) as pose:
    while True:
        ret, frame = video.read()
        
        ## Rotate video
        # frame = cv2.rotate(frame, cv2.ROTATE_180) 

        ## Convert video to RGB
        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False 

        ## Make detection
        results = pose.process(frame)

        ## Recolor
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ## Get landmarks


        ## Render detections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
        

        ## Show video
        cv2.namedWindow('window 1', cv2.WINDOW_NORMAL & cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('window 1', width, height)
        cv2.imshow('window 1', frame)


        if cv2.waitKey(10) & 0xFF == ord('q'):  #pressing q will close the window
            break