import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import mediapipe as mp
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, filtfilt, butter


mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def get_angle(joint, width, height):  # solo utilizo 2D
    pt1 = (joint[0].x * width, joint[0].y * height) 
    pt2 = (joint[1].x * width, joint[1].y * height)
    pt3 = (joint[2].x * width, joint[2].y * height)


    a = np.array(pt3) - np.array(pt2)
    b = np.array(pt1) - np.array(pt2)


    # Calculate the dot product of the two vectors
    dot_prod = np.dot(a, b)

    # Calculate the magnitudes of the two vectors
    mag_vec1 = np.linalg.norm(a)
    mag_vec2 = np.linalg.norm(b)


    # Calculate the angle between the two vectors in radians
    angle_rad = np.arccos(dot_prod / (mag_vec1 * mag_vec2))

    # Convert the angle to degrees
    angle_deg = angle_rad * 180 / np.pi

    return angle_deg


def init_angle_plot(name, video_duration):
    # Initialize the figure and axis for the plot
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Define axis limits
    ax1.set_xlim(0, 3.5)
    ax1.set_ylim(0, 65)
    ax2.set_xlim(0, 3.5)
    ax2.set_ylim(0, 65)

    # Initialize the line object for the plot
    line1, = ax1.plot([], [])
    line2, = ax2.plot([], [])

    # Create an empty list to store the values for the vector
    angle = []
    filt_angle = []

    # Setting title
    plt.suptitle(name)

    # Setting x-axis label and y-axis label
    fig.supxlabel("time (s)")
    fig.supylabel("angle (º)")

    return line1, line2, angle, filt_angle, ax1, ax2


if __name__ == '__main__':
    
    # Load video
    video = cv.VideoCapture('./IMG_9407.mp4')

    # Get duration of video
    frame_count = video.get(cv.CAP_PROP_FRAME_COUNT)
    fps = 30
    video_duration = frame_count/fps

    # Get height and width
    w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Get rescaled height and width for visualization purposes
    scale_percent = 50  # percent of original size
    width = int(w * scale_percent / 100)
    height = int(h * scale_percent / 100)

    # Build plot for knee angle
    rknee_line, filt_line, angle_vect, filt_angle_vect, ax1, ax2 = init_angle_plot('Right Knee Flexion', video_duration)


    # Define extra parameters
    frame_num = 0           # set counter to 0
    filt_angle = []         # start filtered angle vector
    time = []               # start time vector

    # Run MediaPipe Pose
    with mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.9, min_tracking_confidence=0.1) as pose:
        while True:
            # Keep trak of current frame
            frame_num += 1
            # Turn to time (s) 
            new_time = (frame_num/fps)
            time.append(new_time)

            # Read video
            ret, frame = video.read()

            # Process image with MediaPipe Pose model
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame.flags.writeable = False

            results = pose.process(frame)
            
            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            # Extract keypoints of interest (normalized)
            left_knee = [results.pose_landmarks.landmark[i] for i in [24, 26, 28]]

            # Draw results
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

            cv.namedWindow('MediaPipe results', cv.WINDOW_NORMAL & cv.WINDOW_KEEPRATIO)
            cv.resizeWindow('MediaPipe results', width, height)
            cv.imshow('MediaPipe results', frame)

            k = cv.waitKey(10)

            if k == 27:  # Press ESC to quit
                break

            # Generate a new value for the vector
            new_value = 180 - np.array(get_angle(left_knee, w, h))
            angle_vect.append(new_value)

            # Apply Low Pass filter and plot results
            if len(angle_vect) > 15:

                nyquist_freq = 0.5 * fps
                cutoff_freq = 5.0
                b, a = butter(4, cutoff_freq/nyquist_freq, 'low')
                filtered_angle_vect = filtfilt(b, a, angle_vect)

                plt.figure(1)
                filt_line.set_data(time, filtered_angle_vect)
                ax2.figure.canvas.draw()


            # Plot original signal
            plt.figure(1)
            rknee_line.set_data(time, angle_vect)
            ax1.figure.canvas.draw()

            # Save figure
            plt.savefig('mp_angle.png')




