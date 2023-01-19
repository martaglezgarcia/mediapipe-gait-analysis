
# CALIBRATION FUNCTIONS
# https://learnopencv.com/camera-calibration-using-opencv/

import cv2 as cv
import numpy as np
import os
import glob
import sys


# GET INTRINSIC PARAMETERS
def get_intrinsic(camera_id,board):
    #camera_id is the camera name/path to images

    images_names = glob.glob(camera_id)

    # read frames
    images = [cv.imread(imname, 1) for imname in images_names]

    # criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 
    
    
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, board[0] * board[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob(camera_id)

    for frame in images:
        img = cv.imread(frame)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv.findChessboardCorners(gray, board, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        # CALIB_CB_ADAPTIVE_THRESH Use adaptive thresholding to convert the image to black and white
        # CALIB_CB_NORMALIZE_IMAGE Normalize the image gamma with #equalizeHist before applying fixed or adaptive thresholding
        # CALIB_CB_FAST_CHECK Run a fast check on the image that looks for chessboard corners, and shortcut the call if none is found

        if ret == True:

            conv_size = (11,11) # convolution size to improve corner detection

            # refining pixel coordinates for given 2d points
            corners2 = cv.cornerSubPix(gray, corners, conv_size, (-1,-1), criteria)

            # draw and display the corners
            cv.drawChessboardCorners(img, board, corners2, ret)
            cv.putText(img, 'If detected points are poor, press "s" to skip this sample',
                       (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            
            cv.imshow('img',img)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping this sample...')
                continue


            objpoints.append(objp)            
            imgpoints.append(corners2)
    
  
    
    cv.destroyAllWindows()

    # Calibration 

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print('rmse: ', ret)  #should be less than 0.3

    
    return mtx, dist, rvecs, tvecs


# SAVE INTRINSIC PARAMETERS
def save_intrinsic(cam_matrix, dist_coef, camera_id):
    
    # create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')
    
    out_filename = os.path.join('camera_parameters', camera_id + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')

    for i in cam_matrix:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for j in dist_coef[0]:
        outf.write(str(j) + ' ')
    outf.write('\n')


