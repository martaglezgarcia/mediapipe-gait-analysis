
##  Marta González García v6
#   Make detections with the camera real-time
#   Make detections with MediaPipe Pose
#   Get landmarks
#   Save the video

#   It requires the intallation of cv2, mediapipe and numpy packages

                        
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# use webcam
video = cv2.VideoCapture(0)

# get height and width
def get_hw(video):
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return w, h

w,h = get_hw(video)

# get rescaled height and width
scale_percent = 100 # percent of original size
width = int(w * scale_percent / 100)
height = int(h * scale_percent / 100)

# video saved details
result = cv2.VideoWriter('recording.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         20, (w,h))

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
        
        result.write(frame)

        ## Show video
        cv2.namedWindow('window 1', cv2.WINDOW_NORMAL & cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('window 1', width, height)
        cv2.imshow('window 1', frame)


        if cv2.waitKey(10) & 0xFF == ord('q'):  #pressing q will close the window
            break
        


video.release()
result.release()

cv2.destroyAllWindows()


