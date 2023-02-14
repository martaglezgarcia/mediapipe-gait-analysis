#
# LIST OF FUNCTIONS:
#
# CALIBRATION STEP:
#

import math
import cv2 as cv
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import yaml

#___________________CALIBRATION__________________________
# GET INTRINSIC AND EXTRINSIC PARAMETERS FROM CAMERAS
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

        cv.putText(frame0_small, "Press ""s"" to save a frame", (50, 100), font, 0.5, (0, 0, 255), 1)

        k = cv.waitKey(1)

        if k == 27:  # ESC
            break
        elif k == ord('s'):
                savename0 = os.path.join('stereo_frames', cam0_id + '_1.png')
                cv.imwrite(savename0, frame0)
                savename1 = os.path.join('stereo_frames', cam1_id + '_1.png')
                cv.imwrite(savename1, frame1)

                cv.putText(frame0_small, "If you want to exit, press ESC")

        # show both images in the same window
        comb = np.concatenate((frame0_small, frame1_small), axis=1)
        comb = cv.resize(comb, (1398, 476))
        cv.namedWindow("Getting stereo frames ")
        cv.imshow('Getting stereo frames', comb)

        # break out of the loop when the user presses n
        k = cv.waitKey(1)

        if k == 27:  # ESC
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

    img_shape = (calibration_settings['frame_width'], calibration_settings['frame_height'])

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
    new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, img_shape, 1, img_shape)

    new_mtx = np.round(new_mtx, decimals=3)

    print('Individual camera rmse: ', ret)  # should be less than 0.3

    return new_mtx, dist, rvecs, tvecs


def save_intrinsics(cam_id, mtx, dist):
    # create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', cam_id + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('K:\n')
    for i in mtx:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    outf.write('dist:\n')
    for j in dist[0]:
        outf.write(str(j) + ' ')
    outf.write('\n')


# WITH JUST TWO IMAGES
def draw_keypoints_and_match(img1, img2):
    """This function is used for finding keypoints and dercriptors in the image and
        find best matches using brute force/FLANN based matcher."""

    # Note : Can use sift too to improve feature extraction, but it can be patented again so it could brake the code in future!
    # Note: ORB is not scale independent so number of keypoints depend on scale
    # Initiate ORB detector
    orb = cv.ORB_create(nfeatures=500)
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # ________________________Brute Force Matcher_____________________________
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # print(len(matches))

    # Select first 30 matches.
    final_matches = matches[:30]

    # Draw keypoints
    img_with_keypoints = cv.drawMatches(img1, kp1, img2, kp2, final_matches, None,
                                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # create folder if it does not exist
    if not os.path.exists('images'):
        os.mkdir('images')

    savename = os.path.join('images', 'images_with_matching_keypoints.png')
    cv.imwrite(savename, img_with_keypoints)

    # Getting x,y coordinates of the matches
    list_kp1 = [list(kp1[mat.queryIdx].pt) for mat in final_matches]
    list_kp2 = [list(kp2[mat.trainIdx].pt) for mat in final_matches]

    return list_kp1, list_kp2


def calculate_F_matrix(list_kp1, list_kp2):
    """This function is used to calculate the F matrix from a set of 8 points using SVD.
        Furthermore, the rank of F matrix is reduced from 3 to 2 to make the epilines converge."""

    A = np.zeros(shape=(len(list_kp1), 9))

    for i in range(len(list_kp1)):
        x1, y1 = list_kp1[i][0], list_kp1[i][1]
        x2, y2 = list_kp2[i][0], list_kp2[i][1]
        A[i] = np.array([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])

    U, s, Vt = np.linalg.svd(A)
    F = Vt[-1, :]
    F = F.reshape(3, 3)

    # Downgrading the rank of F matrix from 3 to 2
    Uf, Df, Vft = np.linalg.svd(F)
    Df[2] = 0
    s = np.zeros((3, 3))
    for i in range(3):
        s[i][i] = Df[i]

    F = np.dot(Uf, np.dot(s, Vft))
    return F


def RANSAC_F_matrix(list_of_cood_list):
    """This method is used to shortlist the best F matrix using RANSAC based on the number of inliers."""

    list_kp1 = list_of_cood_list[0]
    list_kp2 = list_of_cood_list[1]
    pairs = list(zip(list_kp1, list_kp2))
    max_inliers = 20
    threshold = 0.05  # Tune this value

    for i in range(1000):
        pairs = rd.sample(pairs, 8)
        rd_list_kp1, rd_list_kp2 = zip(*pairs)
        F = calculate_F_matrix(rd_list_kp1, rd_list_kp2)

        tmp_inliers_img1 = []
        tmp_inliers_img2 = []

        for i in range(len(list_kp1)):
            img1_x = np.array([list_kp1[i][0], list_kp1[i][1], 1])
            img2_x = np.array([list_kp2[i][0], list_kp2[i][1], 1])
            distance = abs(np.dot(img2_x.T, np.dot(F, img1_x)))
            # print(distance)

            if distance < threshold:
                tmp_inliers_img1.append(list_kp1[i])
                tmp_inliers_img2.append(list_kp2[i])

        num_of_inliers = len(tmp_inliers_img1)

        # if num_of_inliers > inlier_count:
        #     inlier_count = num_of_inliers
        #     Best_F = F

        if num_of_inliers > max_inliers:
            print("Number of inliers", num_of_inliers)
            max_inliers = num_of_inliers
            Best_F = F
            inliers_img1 = tmp_inliers_img1
            inliers_img2 = tmp_inliers_img2
            # print("Best F matrix", Best_F)

    return Best_F


def calculate_E_matrix(F, K1, K2):
    """Calculation of Essential matrix"""

    E = np.dot(K2.T, np.dot(F, K1))
    return E

# GET ROTATION AND TRANSLATION FROM ESSENTIAL MATRIX
def extract_cameraposes(E):
    """This function extracts all the camera pose solutions from the E matrix"""

    U, s, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    t1, t2 = U[:, 2], -U[:, 2]
    R1, R2 = np.dot(U, np.dot(W, Vt)), np.dot(U, np.dot(W.T, Vt))
    # print("C1", C1, "\n", "C2", C2, "\n", "R1", R1, "\n", "R2", R2, "\n")

    camera_poses = [[R1, t1], [R1, t2], [R2, t1], [R2, t2]]
    return camera_poses


def get_camerapose(camera_poses, list_kp1):
    """This function is used to find the correct camera pose based on the chirelity condition from all 4 solutions."""

    max_len = 0
    # Calculating 3D points
    for pose in camera_poses:

        front_points = []
        for point in list_kp1:
            # Chirelity check
            X = np.array([point[0], point[1], 1])
            V = X - pose[1]

            condition = np.dot(pose[0][2], V)
            if condition > 0:
                front_points.append(point)

        if len(front_points) > max_len:
            max_len = len(front_points)
            best_camera_pose = pose

    return best_camera_pose


def calculate_P_matrix(cmtx, R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t.reshape(3)
    Rt[3, 3] = 1

    P = cmtx @ Rt[:3, :]
    return P


def save_extrinsics(cam_id, R, t):
    # create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', cam_id + '_extrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('R:\n')
    for i in R:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    outf.write('t:\n')
    for i in t:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    print('R and t correctly saved!')


def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift=50.):

        cmtx0 = np.array(camera0_data[0])
        dist0 = np.array(camera0_data[1])
        R0 = np.array(camera0_data[2])
        T0 = np.array(camera0_data[3])
        cmtx1 = np.array(camera1_data[0])
        dist1 = np.array(camera1_data[1])
        R1 = np.array(camera1_data[2])
        T1 = np.array(camera1_data[3])

        P0 = calculate_P_matrix(cmtx0, R0, T0)
        P1 = calculate_P_matrix(cmtx1, R1, T1)

        # define coordinate axes in 3D space. These are just the usual coorindate vectors
        coordinate_points = np.array([[0., 0., 0.],
                                      [1., 0., 0.],
                                      [0., 1., 0.],
                                      [0., 0., 1.]])
        z_shift = np.array([0., 0., _zshift]).reshape((1, 3))
        # increase the size of the coorindate axes and shift in the z direction
        draw_axes_points = 5 * coordinate_points + z_shift

        # project 3D points to each camera view manually. This can also be done using cv.projectPoints()
        # Note that this uses homogenous coordinate formulation
        pixel_points_camera0 = []
        pixel_points_camera1 = []
        for _p in draw_axes_points:
            X = np.array([_p[0], _p[1], _p[2], 1.])

            # project to camera0
            uv = P0 @ X
            uv = np.array([uv[0], uv[1]]) / uv[2]
            pixel_points_camera0.append(uv)

            # project to camera1
            uv = P1 @ X
            uv = np.array([uv[0], uv[1]]) / uv[2]
            pixel_points_camera1.append(uv)

        # these contain the pixel coorindates in each camera view as: (pxl_x, pxl_y)
        pixel_points_camera0 = np.array(pixel_points_camera0)
        pixel_points_camera1 = np.array(pixel_points_camera1)

        # open the video streams
        cap0 = cv.VideoCapture(calibration_settings[camera0_name])
        cap1 = cv.VideoCapture(calibration_settings[camera1_name])

        # set camera resolutions
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

            # follow RGB colors to indicate XYZ axes respectively
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            # draw projections to camera0
            origin = tuple(pixel_points_camera0[0].astype(np.int32))
            for col, _p in zip(colors, pixel_points_camera0[1:]):
                _p = tuple(_p.astype(np.int32))
                cv.line(frame0, origin, _p, col, 2)

            # draw projections to camera1
            origin = tuple(pixel_points_camera1[0].astype(np.int32))
            for col, _p in zip(colors, pixel_points_camera1[1:]):
                _p = tuple(_p.astype(np.int32))
                cv.line(frame1, origin, _p, col, 2)

            cv.imshow('frame0', frame0)
            cv.imshow('frame1', frame1)

            k = cv.waitKey(1)
            if k == 27: break

        cv.destroyAllWindows()

#______________RECTIFICATION AND EPILINES____________________
def rectification(img1, img2, pts1, pts2, F):
    """This function is used to rectify the images to make camera pose's parallel and thus make epiplines as horizontal.
        Since camera distortion parameters are not given we will use cv2.stereoRectifyUncalibrated(), instead of stereoRectify().
    """

    # Stereo rectification
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))
    print(H1)
    print(H2)
    rectified_pts1 = np.zeros(pts1.shape, dtype=int)
    rectified_pts2 = np.zeros(pts2.shape, dtype=int)

    # Rectify the feature points
    for i in range(pts1.shape[0]):
        source1 = np.array([pts1[i][0], pts1[i][1], 1])
        new_point1 = np.dot(H1, source1)
        new_point1[0] = int(new_point1[0] / new_point1[2])
        new_point1[1] = int(new_point1[1] / new_point1[2])
        new_point1 = np.delete(new_point1, 2)
        rectified_pts1[i] = new_point1

        source2 = np.array([pts2[i][0], pts2[i][1], 1])
        new_point2 = np.dot(H2, source2)
        new_point2[0] = int(new_point2[0] / new_point2[2])
        new_point2[1] = int(new_point2[1] / new_point2[2])
        new_point2 = np.delete(new_point2, 2)
        rectified_pts2[i] = new_point2

    # Rectify the images and save them
    img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))

    # create folder if it does not exist
    if not os.path.exists('images'):
        os.mkdir('images')

    savename1 = os.path.join('images', 'rectified_1.png')
    cv.imwrite(savename1, img1_rectified)
    savename2 = os.path.join('images', 'rectified_2.png')
    cv.imwrite(savename2, img2_rectified)

    return rectified_pts1, rectified_pts2, img1_rectified, img2_rectified, H1, H2


def save_rectification(H1, H2):
    # create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', 'rectification.dat')
    outf = open(out_filename, 'w')

    outf.write('H1:\n')
    for i in H1:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    outf.write('H2:\n')
    for i in H2:
        for j in i:
            outf.write(str(j) + ' ')
        outf.write('\n')

    print('Rectification parameters correctly saved!')


def drawlines(img1src, img2src, lines, pts1src, pts2src):
    """This function is used to visualize the epilines on the images
        img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines """
    r, c = img1src.shape
    img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
    img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv.circle(img1color, tuple(pt1), 3, color, -1)
        img2color = cv.circle(img2color, tuple(pt2), 3, color, -1)

    return img1color, img2color

#____________________DISPARITY MAP__________________________

def sum_of_sq_diff(pixel_vals_1, pixel_vals_2):
    """Sum of squared distances for Correspondence"""
    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return np.sum((pixel_vals_1 - pixel_vals_2)**2)


def block_comparison(y, x, block_left, right_array, block_size, x_search_block_size, y_search_block_size):
    """Block comparison function used for comparing windows on left and right images and find the minimum value ssd match the pixels"""

    # Get search range for the right image
    x_min = max(0, x - x_search_block_size)
    x_max = min(right_array.shape[1], x + x_search_block_size)
    y_min = max(0, y - y_search_block_size)
    y_max = min(right_array.shape[0], y + y_search_block_size)

    first = True
    min_ssd = None
    min_index = None

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            block_right = right_array[y: y + block_size, x: x + block_size]
            ssd = sum_of_sq_diff(block_left, block_right)
            if first:
                min_ssd = ssd
                min_index = (y, x)
                first = False
            else:
                if ssd < min_ssd:
                    min_ssd = ssd
                    min_index = (y, x)

    return min_index


def get_disparity(img1, img2):
    block_size = 15  # 15
    x_search_block_size = 50  # 50
    y_search_block_size = 1
    h, w = img1.shape
    disparity_map = np.zeros((h, w))

    for y in range(block_size, h - block_size):
        for x in range(block_size, w - block_size):
            block_left = img1[y:y + block_size, x:x + block_size]
            index = block_comparison(y, x, block_left, img2, block_size, x_search_block_size, y_search_block_size)
            disparity_map[y, x] = abs(index[1] - x)

    disparity_map_unscaled = disparity_map.copy()

    # Scaling the disparity map
    max_pixel = np.max(disparity_map)
    min_pixel = np.min(disparity_map)

    for i in range(disparity_map.shape[0]):
        for j in range(disparity_map.shape[1]):
            disparity_map[i][j] = int((disparity_map[i][j] * 255) / (max_pixel - min_pixel))

    disparity_map_scaled = disparity_map
    return disparity_map_unscaled, disparity_map_scaled



