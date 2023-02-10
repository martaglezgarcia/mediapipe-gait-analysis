

import math
import cv2 as cv
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import yaml
from scipy import linalg
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#______________________GET CAMERA PARAMETERS________________________
def read_intrinsics(camera_id):
    inf = open('./camera_parameters/' + str(camera_id) + '_intrinsics.dat', 'r')
    K = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        K.append(line)
    line = inf.readline()
    line = inf.readline().split()
    line = line = [float(en) for en in line]
    dist.append(line)
    inf.close()
    return np.array(K), np.array(dist)


def read_extrinsics(camera_id):
    inf = open('./camera_parameters/' + str(camera_id) + '_extrinsics.dat', 'r')
    R = []
    t = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        R.append(line)
    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        t.append(line)
    inf.close()
    return np.array(R), np.array(t)


def read_rectification():
    inf = open('./camera_parameters/' + 'rectification.dat', 'r')
    H1 = []
    H2 = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        H1.append(line)
    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        H2.append(line)
    inf.close()
    return np.array(H1), np.array(H2)


def calculate_P_matrix(camera_id):

    # read camera parameters
    K, dist = read_intrinsics(camera_id)
    R, t = read_extrinsics(camera_id)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t.reshape(3)
    Rt[3, 3] = 1

    P = K @ Rt[:3, :]
    return P


def convert_pt_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis=1)
    else:
        return np.cocatenate([pts, [1]], axis=0)


def DLT(P1, P2, point1, point2):
    # The method consists of constructing a matrix A from the input data,
    # computing the singular value decomposition of the product of the
    # transpose of A and A, and finally returning the last row of the V matrix,
    # divided by its last component, as the estimated 3D point

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    # Using SVD to get a pseudo-inverse of P to get the 3D points
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3, 0:3]/Vh[3, 3]


def get_hw(video):
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    return width, height


#__________________________MEDIA PIPE___________________________
def visualize_3d(p3ds):

    """Now visualize in 3D"""
    torso = [[0, 1] , [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    body = [torso, arml, armr, legr, legl]
    colors = ['red', 'blue', 'green', 'black', 'orange']

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for framenum, kpts3d in enumerate(p3ds):
        if framenum%2 == 0: continue #skip every 2nd frame
        for bodypart, part_color in zip(body, colors):
            for _c in bodypart:
                ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)

        #uncomment these if you want scatter plot of keypoints and their indices.
        # for i in range(12):
        #     #ax.text(kpts3d[i,0], kpts3d[i,1], kpts3d[i,2], str(i))
        #     #ax.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])


        #ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(-10, 10)
        ax.set_xlabel('x')
        ax.set_ylim3d(-10, 10)
        ax.set_ylabel('y')
        ax.set_zlim3d(-10, 10)
        ax.set_zlabel('z')
        plt.pause(0.1)
        ax.cla()

def run_mp(v0, v1, keypoints, P1, P2):

    vids = [v0, v1]

    # get height and width
    w0, h0 = get_hw(v0)
    w1, h1 = get_hw(v1)

    # MEDIAPIPE POSE
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as pose:
        while True:
            ret0, frame0 = v0.read()
            ret1, frame1 = v1.read()

            # Exit if no camera input
            if not ret0 or not ret1:
                break

            # Convert videos to RGB
            frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
            frame0.flags.writeable = False  # improve performance
            frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
            frame1.flags.writeable = False

            # Make detections
            results0 = pose.process(frame0)
            results1 = pose.process(frame1)

            # Recolor
            frame0.flags.writeable = True
            frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
            frame1.flags.writeable = True
            frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

            # Get landmarks
            def detect_keypoints(frame, results, keypoints):
                frame_kpts = []
                if results.pose_landmarks:
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        if i not in keypoints:
                            continue # save keypoints that are in keypoints vector
                        pxl_x = int(round(landmark.x * frame.shape[1]))
                        pxl_y = int(round(landmark.y * frame.shape[0]))
                        cv.circle(frame, (pxl_x, pxl_y), 3, (0,0,255), -1)  # draw keypoint
                        kpts = [pxl_x, pxl_y]
                        frame_kpts.append(kpts)
                    else: # add [-1,-1] if no keypoints are found
                        frame_kpts = [[-1,-1]] * len(keypoints)

                return frame_kpts

            frame0_kpts = detect_keypoints(frame0, results0, keypoints)
            frame1_kpts = detect_keypoints(frame1, results1, keypoints)

            # Calculate 3D position
            frame_p3ds = []
            for uv1, uv2 in zip(frame0_kpts, frame1_kpts):
                if uv1[0] == -1 or uv2[0] == -1:
                    p3d = [-1, -1, -1]
                else:
                    p3d = DLT(P1, P2, uv1, uv2)
                frame_p3ds.append(p3d)

            frame_p3ds = np.array(frame_p3ds).reshape(12, 3)

            # Spatiotemporal parameters' calculation (just knee flexion)





            # Render detections
            mp_drawing.draw_landmarks(frame0, results0.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

            mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=3),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

            # Show videos
            cv.namedWindow('window 1', cv.WINDOW_NORMAL & cv.WINDOW_KEEPRATIO)
            cv.resizeWindow('window 1', w0, h0)
            cv.imshow('window 1', frame0)

            cv.namedWindow('window 2', cv.WINDOW_NORMAL & cv.WINDOW_KEEPRATIO)
            cv.resizeWindow('window 2', w1, h1)
            cv.imshow('window 2', frame1)

            k = cv.waitKey(1)

            if k & 0xFF == 27:  # pressing ESC will close the windows
                break

        cv.destroyAllWindows()
        for vid in vids:
            vid.release()

        return np.array(kpts_3d)

