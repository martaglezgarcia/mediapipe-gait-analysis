#
# This file contains a list of functions that will be called from the main:
#
#   FUNCTIONS:  - load_calibration_settings(filename)
#               - take_mono_frames(cam_id)
#               - take_stereo_frames(cam0_id,cam1_id)
#               - get_intrinsics(cam_path)
#               - save_intrinsics(mtx, dist, cam_id)
#               - stereo_cal(stcam0_path, stcam1_path, mtx0, dist0, mtx1, dist1)
#               - save_stereo_cal(R, T, E, F, rect0, rect1, P0, P1, Q, stereoMap0, stereoMap1)


import cv2 as cv
import numpy as np
import os
import glob
import sys
import yaml

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
    return calibration_settings


def take_mono_frames(cam_id):
    # create directory if it doesn't exist already
    if not os.path.exists('single_frames'):
        os.mkdir('single_frames')

    # settings from YAML file
    resize = calibration_settings['view_resize']
    max_frames = calibration_settings['mono_cal_frames']

    # open camera

    cap = cv.VideoCapture(calibration_settings[cam_id])

    # set resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap.set(3, width)
    cap.set(4, height)

    # additional
    cur_frame = 1
    font = cv.FONT_HERSHEY_COMPLEX

    while cap.isOpened():

        ret, frame = cap.read()
        frame = cv.flip(frame, 1)  # mirror camera

        if not ret:
            print('Camera not returning video data. Exiting...')
            quit()

        frame_small = cv.resize(frame, None, fx=1. / resize, fy=1. / resize)

        cv.putText(frame_small, "Make sure the camera shows the calibration pattern well", (50, 50), font, 0.5,
                   (0, 0, 255), 1)
        cv.putText(frame_small, "Press ""s"" to save a frame", (50, 100), font, 0.5, (0, 0, 255), 1)

        k = cv.waitKey(1)

        if k == 27:  # ESC
            break
        elif k == ord('s'):
            savename = os.path.join('single_frames', cam_id + '_' + str(cur_frame) + '.png')
            cv.imwrite(savename, frame)

            print('Image saved! You have taken:', str(cur_frame), 'frames')
            cur_frame += 1

        cv.namedWindow(cam_id)
        cv.imshow(cam_id, frame_small)

        # break out of the loop when enough number of frames have been saved
        if cur_frame == max_frames + 1:
            break

    cv.destroyAllWindows()


def take_stereo_frames(cam0_id, cam1_id):
    # create directory if it doesn't exist already
    if not os.path.exists('stereo_frames'):
        os.mkdir('stereo_frames')

    # settings from YAML file
    resize = calibration_settings['view_resize']
    max_frames = calibration_settings['stereo_cal_frames']

    # open cameras
    cap0 = cv.VideoCapture(calibration_settings[cam0_id])
    cap1 = cv.VideoCapture(calibration_settings[cam1_id])

    # set resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    # additional
    cur_frame = 1
    font = cv.FONT_HERSHEY_COMPLEX

    while cap0.isOpened():

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        # mirror cameras
        frame0 = cv.flip(frame0, 1)
        frame1 = cv.flip(frame1, 1)

        if not ret0 or not ret1:
            print('Cameras not returning video data. Exiting...')
            quit()

        frame0_small = cv.resize(frame0, None, fx=1. / resize, fy=1. / resize)
        frame1_small = cv.resize(frame1, None, fx=1. / resize, fy=1. / resize)

        cv.putText(frame0_small, "Make sure the camera shows the calibration pattern well", (50, 50), font, 0.5,
                   (0, 0, 255), 1)
        cv.putText(frame0_small, "Press ""s"" to save a frame", (50, 100), font, 0.5, (0, 0, 255), 1)

        k = cv.waitKey(1)

        if k == 27:  # ESC
            break
        elif k == ord('s'):
            savename0 = os.path.join('stereo_frames', cam0_id + '_' + str(cur_frame) + '.png')
            cv.imwrite(savename0, frame0)
            savename1 = os.path.join('stereo_frames', cam1_id + '_' + str(cur_frame) + '.png')
            cv.imwrite(savename1, frame1)

            cur_frame += 1
            cv.putText(frame0_small, "Current frame: " + str(cur_frame), (50, 100), font, 1, (0, 255, 0), 1)

        # show both images in the same window
        comb = np.concatenate((frame0_small, frame1_small), axis=1)
        comb = cv.resize(comb, (1398, 476))
        cv.namedWindow("Getting stereo frames ")
        cv.imshow('Getting stereo frames', comb)

        # break out of the loop when enough number of frames have been saved
        if cur_frame == max_frames + 1:
            break

    cv.destroyAllWindows()


def get_intrinsics(cam_path):
    images_names = glob.glob(cam_path)

    # read frames
    images = [cv.imread(imname, 1) for imname in images_names]

    # criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    rows = calibration_settings['board_rows']
    columns = calibration_settings['board_cols']
    board = (rows, columns)

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, board[0] * board[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)

    # Extracting path of individual image stored in a given directory
    images = glob.glob(cam_path)

    for frame in images:
        img = cv.imread(frame)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv.findChessboardCorners(gray, board,
                                                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        # CALIB_CB_ADAPTIVE_THRESH Use adaptive thresholding to convert the image to black and white
        # CALIB_CB_NORMALIZE_IMAGE Normalize the image gamma with #equalizeHist before applying fixed or adaptive thresholding
        # CALIB_CB_FAST_CHECK Run a fast check on the image that looks for chessboard corners, and shortcut the call if none is found

        if ret == True:

            conv_size = (11, 11)  # convolution size to improve corner detection

            # refining pixel coordinates for given 2d points
            corners2 = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)

            # draw and display the corners
            cv.drawChessboardCorners(img, board, corners2, ret)
            cv.putText(img, 'If detected points are poor, press "s" to skip this sample',
                       (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            cv.imshow("img", img)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping this sample...')
                continue

            objpoints.append(objp)
            imgpoints.append(corners2)

    cv.destroyAllWindows()

    # Calibration

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print('Individual camera rmse: ', ret)  # should be less than 0.3

    return mtx, dist, rvecs, tvecs


def save_intrinsics(mtx, dist, cam_id):
    # create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', cam_id + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for i in mtx:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for j in dist[0]:
        outf.write(str(j) + ' ')
    outf.write('\n')


def stereo_cal(stcam0_path, stcam1_path, mtx0, dist0, mtx1, dist1):
    # read the synched frames and sort in order
    c0_names = sorted(glob.glob(stcam0_path))
    c1_names = sorted(glob.glob(stcam1_path))

    # open images
    c0_images = [cv.imread(imname, 1) for imname in c0_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_names]

    # change this if stereo calibration not good
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # calibration pattern settings
    rows = calibration_settings['board_rows']
    columns = calibration_settings['board_cols']
    world_scaling = calibration_settings['square_size']

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    # check if they are the same size
    if c0_images[0].shape[1] == c1_images[0].shape[1] and c0_images[0].shape[0] == c1_images[0].shape[0]:
        width = c0_images[0].shape[1]
        height = c0_images[0].shape[0]
    else:
        print('Frames are not the same size')
        quit()

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    count = 0

    for frame0, frame1 in zip(c0_images, c1_images):
        gray0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret0, corners0 = cv.findChessboardCorners(gray0, (rows, columns), None)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)

        if c_ret0 and c_ret1 == True:

            corners0 = cv.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)

            p0_c0 = corners0[0, 0].astype(np.int32)
            p0_c1 = corners1[0, 0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c0[0], p0_c0[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.drawChessboardCorners(frame0, (rows, columns), corners0, c_ret0)

            cv.putText(frame1, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)

            combo = np.concatenate((frame0, frame1), axis=1)
            combo = cv.resize(combo, (1398, 476))
            cv.namedWindow("Stereo images window")
            cv.imshow("Stereo images window", combo)

            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping these samples...')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners0)
            imgpoints_right.append(corners1)
            count += 1

    cv.destroyAllWindows()

    # Calibration part
    new_mtx0, roi0 = cv.getOptimalNewCameraMatrix(mtx0, dist0, (width, height), 1, (width, height))
    new_mtx1, roi1 = cv.getOptimalNewCameraMatrix(mtx1, dist1, (width, height), 1, (width, height))

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, K1, D1, K2, D2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, new_mtx0, dist0,
                                                         new_mtx1, dist1, (width, height), criteria=criteria,
                                                         flags=stereocalibration_flags)
    print(str(count) + ' images were analyzed')
    print('Stereo calibration rmse: ', ret)

    # Rectification part
    rectifyScale = 1
    rect0, rect1, P0, P1, Q, roi0, roi1 = cv.stereoRectify(new_mtx0, dist0, new_mtx1, dist1,
                                                           gray0.shape[::-1], R, T, rectifyScale)
    stereoMap0 = cv.initUndistortRectifyMap(new_mtx0, dist0, rect0, P0, gray0.shape[::-1], cv.CV_16SC2)
    stereoMap1 = cv.initUndistortRectifyMap(new_mtx1, dist1, rect1, P1, gray1.shape[::-1], cv.CV_16SC2)

    return R, T, E, F, rect0, rect1, P0, P1, Q, stereoMap0, stereoMap1


def save_stereo_cal(R, T, E, F, rect0, rect1, P0, P1, Q, stereoMap0, stereoMap1):
    # create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', 'extrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('R:\n')
    for i in R:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for i in T:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    outf.write('E:\n')
    for i in E:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    outf.write('F:\n')
    for i in R:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')
    print('R, T, E, and F correctly saved!')

    #   RECTIFICATION DATA
    out_filename = os.path.join('camera_parameters', 'rectification.dat')
    outf = open(out_filename, 'w')

    outf.write('P0:\n')
    for i in P0:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    outf.write('P1:\n')
    for i in P1:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    outf.write('Q:\n')
    for i in Q:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')
    print('Rectification data correctly saved!')

    #   STEREO MAPS
    out_filename = os.path.join('camera_parameters', 'stereomaps.dat')
    outf = open(out_filename, 'w')

    outf.write('stereoMap0_x:\n')
    for i in stereoMap0[0]:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')
    outf.write('stereoMap0_y:\n')
    for i in stereoMap0[1]:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    outf.write('stereoMap1_x:\n')
    for i in stereoMap1[0]:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')
    outf.write('stereoMap1_y:\n')
    for i in stereoMap1[1]:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')
    print('Stereo-maps correctly saved!')


if __name__ == '__main__':

    calibration_settings = load_calibration_settings('calibration_settings.yaml')
    img_size = [calibration_settings['frame_width'], calibration_settings['frame_height']]

    print('The calibration process for individual cameras will start soon...')

    # STEP 1 - GET CALIBRATION FRAMES FOR INDIVIDUAL CAMERAS
    #take_mono_frames('camera0')
    #take_mono_frames('camera1')

    # STEP 2 - ESTIMATE AND SAVE INTRINSIC PARAMETERS FOR BOTH CAMERAS
    mtx0, dist0, _, _ = get_intrinsics('./single_frames/camera0*')
    save_intrinsics(mtx0, dist0, 'camera0')

    mtx1, dist1, _, _ = get_intrinsics('./single_frames/camera1*')
    save_intrinsics(mtx1, dist1, 'camera1')

    print('Calibration for individual cameras completed.')
    print('Wait till both cameras open to start stereo-calibration.')

    # STEP 3 - GET STEREO FRAMES FOR BOTH CAMERAS
    take_stereo_frames('camera0', 'camera1')

    print('Stereo-calibration completed.')

    # STEP 4 - STEREO CALIBRATION AND RECTIFICATION
    frames_c0 = os.path.join('stereo_frames', 'camera0*')
    frames_c1 = os.path.join('stereo_frames', 'camera1*')

    R, T, E, F, rect0, rect1, P0, P1, Q, stereoMap0, stereoMap1 = stereo_cal(frames_c0, frames_c1,
                                                                                   mtx0, dist0,
                                                                                   mtx1, dist1)
    # save_stereo_cal(R, T, E, F, rect0, rect1, P0, P1, Q, stereoMap0, stereoMap1)

    # STEREOVISION
    cap0 = cv.VideoCapture(calibration_settings['camera0'])
    cap1 = cv.VideoCapture(calibration_settings['camera1'])

    while cap0.isOpened() and cap1.isOpened():
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Camera not returning video data. Exiting...')
            quit()

        # Undistortion and rectification
        frame0 = cv.remap(frame0, stereoMap0[0], stereoMap0[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        frame1 = cv.remap(frame1, stereoMap1[0], stereoMap1[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

        comb = np.concatenate((frame0, frame1), axis=1)
        comb = cv.resize(comb, (1398, 476))
        cv.imshow("Real-time images rectified", comb)

        if cv.waitKey(1) & 0xFF == 27:  # Press ESC to quit
            break

    cap0.release()
    cap1.release()

    cv.destroyAllWindows()

