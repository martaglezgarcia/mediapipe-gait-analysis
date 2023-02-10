

import math
import cv2 as cv
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import yaml

from mp_tools import *

if __name__ == '__main__':
    # STEP 0: COMPUTE THE PROJECTION MATRICES FOR EACH CAMERA
    P1 = calculate_P_matrix('camera0')
    P2 = calculate_P_matrix('camera1')

    # STEP 1: OPEN MEDIAPIPE
    # Open cameras
    v1 = cv.VideoCapture(2)
    v2 = cv.VideoCapture(0)
    # Get height and width
    from mp_tools import get_hw
    w0, h0 = get_hw(v1)
    w1, h1 = get_hw(v2)

    # RECTIFICATION OF THE VIDEOS
    H1, H2 = read_rectification()

    v1_rect = cv.warpPerspective(v1, H1, (w1, h1))
    v2_rect = cv.warpPerspective(v2, H2, (w1, h1))

    # Landmarks
    keypoints = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    # STEP 1 - Run MediaPipe
    run_mp(v1_rect, v2_rect, P1, P2)


