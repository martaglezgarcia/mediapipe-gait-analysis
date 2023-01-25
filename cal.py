
# CALIBRATION FUNCTIONS

# https://learnopencv.com/camera-calibration-using-opencv/
# https://github.com/TemugeB/python_stereo_camera_calibrate 

# functions found in the file:
#       - load_calibration_settings
#       - take_mono_frames
#       - take_stereo_frames
#       - get_intrinsic
#       - save_intrinsic


import cv2 as cv
import numpy as np
import os
import glob 
import sys
import yaml

# ____________________________________________________________________________________
#  
# LOAD CALIBRATION_SETTINGS.YAML
# ____________________________________________________________________________________

calibration_settings = {}

def load_calibration_settings(filename):    

    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    
    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        global calibration_settings
        calibration_settings = yaml.safe_load(f)

# ____________________________________________________________________________________
#  
# TAKE FRAMES FROM SINGLE CAMERAS
# ____________________________________________________________________________________

def take_mono_frames(cam_id):
    

    # create directory if it doesn't exist already
    if not os.path.exists('single_frames'):
        os.mkdir('single_frames')

    # settings from YAML file
    resize = calibration_settings['view_resize']
    max_time = calibration_settings['timer']    
    max_frames = calibration_settings['mono_cal_frames']

    # open cameras
    cap = cv.VideoCapture(calibration_settings[cam_id])

    # set resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap.set(3, width)
    cap.set(4, height)


	# additional
    time = max_time
    start = False
    cur_frame = 1
    font = cv.FONT_HERSHEY_COMPLEX

    while True:

        ret, frame = cap.read()

        if not ret:
            print('Camera not returning video data. Exiting...')
            quit()

        frame_small = cv.resize(frame, None, fx=1./resize, fy=1./resize)

        if not start:
            cv.putText(frame_small, "Make sure the camera shows the calibration pattern well", (50,50), font, 1, (0,0,255), 1)
            cv.putText(frame_small, "Press SPACEBAR to start collecting frames", (50,100), font, 1, (0,0,255), 1)
        
        if start:
            time -= 1
            secs = str(float(time/10))
            cv.putText(frame_small, "Timer: " + str(secs), (50,50), font, 1, (0,255,0), 1)
            cv.putText(frame_small, "Current frame: " + str(cur_frame), (50,100), font, 1, (0,255,0), 1)
            

            #save the frame when timer reaches 0.
            if time <= 0:
                savename = os.path.join('single_frames', cam_id + '_' + str(cur_frame) + '.png')
                cv.imwrite(savename, frame)

                cur_frame += 1
                time = max_time

        cv.imshow(cam_id, frame_small)
        k = cv.waitKey(1)
        
        if k == 27:
            #if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            #Press spacebar to start data collection
            start = True

        #break out of the loop when enough number of frames have been saved
        if cur_frame == max_frames + 1: break

    cv.destroyAllWindows()


# ____________________________________________________________________________________
#
# TAKE CALIBRATION FRAMES FROM BOTH CAMERAS AT THE SAME TIME
# ____________________________________________________________________________________

def take_stereo_frames(cam0, cam1):

    # create directory if it doesn't exist already
    if not os.path.exists('stereo_frames'):
        os.mkdir('stereo_frames')

    # settings from YAML file
    resize = calibration_settings['view_resize']
    max_time = calibration_settings['timer']    
    max_frames = calibration_settings['stereo_cal_frames']

    # open cameras
    cap0 = cv.VideoCapture(calibration_settings[cam0])
    cap1 = cv.VideoCapture(calibration_settings[cam1])

    # set resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

	# additional
    time = max_time
    start = False
    cur_frame = 1
    font = cv.FONT_HERSHEY_COMPLEX

    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Cameras not returning video data. Exiting...')
            quit()

        frame0_small = cv.resize(frame0, None, fx=1./resize, fy=1./resize)
        frame1_small = cv.resize(frame1, None, fx=1./resize, fy=1./resize)

        if not start:
            cv.putText(frame0_small, "Make sure both cameras can see the calibration pattern well", (50,50), font, 1, (0,0,255), 1)
            cv.putText(frame0_small, "Press SPACEBAR to start collection frames", (50,100), font, 1, (0,0,255), 1)
        
        if start:
            time -= 1
            secs = str(float(time/10))
            cv.putText(frame0_small, "Timer: " + str(secs), (50,50), font, 1, (0,255,0), 1)
            cv.putText(frame0_small, "Current frame: " + str(cur_frame), (50,100), font, 1, (0,255,0), 1)
            
            cv.putText(frame1_small, "Timer: " + str(secs), (50,50), font, 1, (0,255,0), 1)
            cv.putText(frame1_small, "Current frame: " + str(cur_frame), (50,100), font, 1, (0,255,0), 1)

            #save the frame when timer reaches 0.
            if time <= 0:
                savename = os.path.join('stereo_frames', cam0 + '_' + str(cur_frame) + '.png')
                cv.imwrite(savename, frame0)

                savename = os.path.join('stereo_frames', cam1 + '_' + str(cur_frame) + '.png')
                cv.imwrite(savename, frame1)

                cur_frame += 1
                time = max_time

        cv.imshow('frame0_small', frame0_small)
        cv.imshow('frame1_small', frame1_small)
        k = cv.waitKey(1)
        
        if k == 27:
            #if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            #Press spacebar to start data collection
            start = True

        #break out of the loop when enough number of frames have been saved
        if cur_frame == max_frames + 1: break

    cv.destroyAllWindows()


# ____________________________________________________________________________________
#
# GET INTRINSIC PARAMETERS
# ____________________________________________________________________________________

def get_intrinsic(camera_id):
    # camera_id is the camera id/path to images

    images_names = glob.glob(camera_id)

    # read frames
    images = [cv.imread(imname, 1) for imname in images_names]

    # criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 
    
    rows = calibration_settings.get('board_rows')
    cols = calibration_settings.get('board_cols')
    board = (rows,cols)

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, board[0] * board[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)

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


# ____________________________________________________________________________________
#
# SAVE INTRINSIC PARAMETERS
# ____________________________________________________________________________________

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



