
# Calibration

import cv2 as cv
import numpy as np
import os
import glob 
import sys
import yaml

import cal

cal.load_calibration_settings('calibration_settings.yaml')

# STEP 2 - GET CALIBRATION FRAMES FOR INDIVIUAL CAMERAS
# cal.take_mono_frames('camera0')
# cal.take_mono_frames('camera1')

# STEP 3 - GET AND SAVE INTRINSIC PARAMETERS FOR INDIVIDUAL CAMERAS
mtx0, dist0, rvecs0, tvecs0 = cal.get_intrinsic('./single_frames/camera0*')
cal.save_intrinsic(mtx0, dist0, 'camera0')

mtx1, dist1, rvecs1, tvecs1 = cal.get_intrinsic('./single_frames/camera1*')
cal.save_intrinsic(mtx1, dist1, 'camera1')

# STEP 4 - Take frames from both cameras at the same time
# cal.take_stereo_frames('camera0', 'camera1')

# STEP 5 - GET ROTATION AND TRANSLATION MATRICES
frames_c0 = os.path.join('stereo_frames', 'camera0*')
frames_c1 = os.path.join('stereo_frames', 'camera1*')

R, T = cal.stereo_cal(mtx0, dist0, mtx1, mtx1, frames_c0, frames_c1)

# STEP 6 - SAVE EXTRINSIC PARAMETERS
# world coordinates are the same as camera0 coordinates
R0 = np.eye(3, dtype=np.float32)
T0 = np.array([0., 0., 0.]).reshape((3, 1))

cal.save_extrinsic(R0, T0, R, T) #this will write R and T to disk
R1 = R; T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1

# STEP 7 - CHECK CALIBRATION MAKES SENSE
camera0_data = [mtx0, dist0, R0, T0]
camera1_data = [mtx1, dist1, R1, T1]

cal.check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift = 60.)
