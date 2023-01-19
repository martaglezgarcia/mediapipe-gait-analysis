
# Calibration

import cv2
import numpy as np
import os
import glob

from calibration_functions import *

board1 = (7,7)
mtx, dist, rvecs, tvecs = get_intrinsic('./cam0/*.jpg',board1)
save_intrinsic(mtx, dist, 'cam0')

board2 = (7,7)
mtx2, dist2, rvecs2, tvecs2 = get_intrinsic('./cam1/*.jpg',board2)
save_intrinsic(mtx2, dist2, 'cam1')