
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
#cal.take_mono_frames('camera0')
#cal.take_mono_frames('camera1')

# STEP 3 - GET AND SAVE INTRINSIC PARAMETERS FOR INDIVIDUAL CAMERAS
mtx, dist, rvecs, tvecs = cal.get_intrinsic('./single_frames/camera0*')
cal.save_intrinsic(mtx, dist, 'camera0')

mtx2, dist2, rvecs2, tvecs2 = cal.get_intrinsic('./single_frames/camera1*')
cal.save_intrinsic(mtx2, dist2, 'camera1')

# STEP 4 - Take frames from both cameras at the same time
cal.take_stereo_frames('camera0', 'camera1')

