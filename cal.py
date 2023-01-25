
# CALIBRATION FUNCTIONS

# https://learnopencv.com/camera-calibration-using-opencv/
# https://github.com/TemugeB/python_stereo_camera_calibrate 

# functions found in the file:
#       - load_calibration_settings
#       - take_mono_frames
#       - take_stereo_frames
#       - get_intrinsic
#       - save_intrinsic
#       - save_extrinsic
#       - get_projection_matrix
#       - check_calibration

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

# ____________________________________________________________________________________
#
# GET ROTATION AND TRANSLATION MATRICES FROM STEREO_FRAMES
# ____________________________________________________________________________________


def stereo_cal(mtx0, dist0, mtx1, dist1, frames_c0, frames_c1):

    # read the synched frames and sort in order
    c0_names = sorted(glob.glob(frames_c0))
    c1_names = sorted(glob.glob(frames_c1))

    # open images
    c0_images = [cv.imread(imname, 1) for imname in c0_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_names]

    # change this if stereo calibration not good
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # calibration pattern settings
    rows = calibration_settings['board_rows']
    columns = calibration_settings['board_cols']
    world_scaling = calibration_settings['square_size']

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    # frame dimensions. Frames should be the same size.
    # check if they are the same size
    if c0_images[0].shape[1] == c1_images[0].shape[1] and c0_images[0].shape[0] == c1_images[0].shape[0]:
        width = c0_images[0].shape[1]
        height = c0_images[0].shape[0]
    else:
        print('Frames are not the same size')
        quit()


    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv.imshow('img', frame0)

            cv.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            cv.imshow('img2', frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria = criteria, flags = stereocalibration_flags)

    print('rmse: ', ret)
    cv.destroyAllWindows()
    return R, T



# ____________________________________________________________________________________
#
# GET PROJECTION MATRIX
# making R and T homogeneous
# ____________________________________________________________________________________

def get_projection_matrix(k, R, T):
    RT = np.zeros((4,4))
    RT[:3,:3] = R
    RT[:3, 3] = T.reshape(3)
    RT[3,3] = 1

    P = k @ RT[:3,:] # @ matrix multiplication
    return P

# ____________________________________________________________________________________
#
# SAVE EXTRINSIC PARAMETERS IN CAMERA_PARAMETERS FOLDER
# ____________________________________________________________________________________

def save_extrinsic(R0, T0, R1, T1, prefix = ''):
#create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    #R1 and T1 are just stereo calibration returned values
    camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R0, T0, R1, T1


# ____________________________________________________________________________________
#
# CHECK CALIBRATION
# ____________________________________________________________________________________

def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift = 50.):
    
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    #define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    #increase the size of the coorindate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points + z_shift

    #project 3D points to each camera view manually. This can also be done using cv.projectPoints()
    #Note that this uses homogenous coordinate formulation
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])
        
        #project to camera0
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera0.append(uv)

        #project to camera1
        uv = P1 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera1.append(uv)

    #these contain the pixel coorindates in each camera view as: (pxl_x, pxl_y)
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    #open the video streams
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    #set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Video stream not returning frame data')
            quit()

        #follow RGB colors to indicate XYZ axes respectively
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        #draw projections to camera0
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame0, origin, _p, col, 2)
        
        #draw projections to camera1
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame1, origin, _p, col, 2)

        cv.imshow('frame0', frame0)
        cv.imshow('frame1', frame1)

        k = cv.waitKey(1)
        if k == 27: break

    cv.destroyAllWindows()