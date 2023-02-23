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
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

from calib_tools import *

if __name__ == '__main__':

    # STEP 0 : LOAD CALIBRATION SETTINGS
    calibration_settings = load_calibration_settings('calibration_settings.yaml')

    # ___________________ INTRINSIC PARAMETERS _______________________
    # Get calibration frames for individual cameras and estimate and
    # save their intrinsic parameters
    #
    print('\nThe first step of the calibration process is to get the intrinsic parameters '
          'of both cameras. \nTo do so, the specified number of mono frames on the settings file '
          'will be taken and\nsaved on the folder "mono_frames". Then, the intrinsic parameters '
          'of the cameras\nwill be estimated. Do you want to skip this step? (y/n)\n')
    ok = input('Answer: ')
    while ok not in ["n", "N", "y", "Y"]:
        print('Please, introduce a valid command:')
        ok = input('Answer: ')
    if ok == "n" or ok == "N":
        # CAMERA 0 ______________________________________________________
        while ok == "n" or ok == "N":
            print('\n\033[4mESTIMATING INTRINSIC PARAMETERS FOR THE FIRST CAMERA\033[0m')
            print('\nDo you want to take new pictures for camera0? (y/n)\n')
            pics = input('Answer: ')
            while pics not in ["n", "N", "y", "Y"]:
                print('Please, introduce a valid command:')
                pics = input('Answer: ')
            if pics == "y" or pics == "Y":
                print('\nOpening the first camera...')
                print('-> Make sure that the checkerboard pattern can be seen by the camera.\n'
                      '-> To ensure a good calibration, take pictures from different angles\n'
                      'Press "s" to save a frame')
                take_mono_frames('camera0')
            print('\nA new window should have opened showing the checkerboard pattern with the detected corners.\n'
                  'If the detections are poor, skip the sample by pressing "s".\n'
                  'Otherwise, press any other key.\n'
                  'Anytime you want to close any emergent window, press ESC\n')
            K1, dist1, _, _ = get_intrinsics('./mono_frames/camera0*')
            print('Do you want to repeat the samples? (y/n)\n'
                  '(You should aim for a RMSE < 0.5)\n')
            ok = input('Answer: ')
            if ok == "N" or ok == "n":
                break
            elif ok not in ["n", "N", "y", "Y"]:
                print('Please, introduce a valid command:')
                ok = input('Answer: ')
        print('___' * 25, '\n')
        print("K matrix =\n", K1)
        print("\nDistortion coefficients =\n", dist1)

        # Save parameters
        save_intrinsics('camera0', K1, dist1)
        print('\nIntrinsic parameters correctly saved for the first camera in "camera_parameters" folder.')
        print('===' * 25, '\n')

        # CAMERA 1 ________________________________________________________
        ok = "n"
        while ok == "n" or ok == "N":
            print('\033[4mESTIMATING INTRINSIC PARAMETERS FOR THE SECOND CAMERA\033[0m')
            print('\nDo you want to take new pictures for camera1? (y/n)\n')
            pics = input('Answer: ')
            if pics not in ["n", "N", "y", "Y"]:
                print('Please, introduce a valid command:')
                pics = input('Answer: ')
            elif pics == "y" or pics == "Y":
                print('\nOpening the second camera...')
                print('-> Make sure that the checkerboard pattern can be seen by the camera.\n'
                      '-> To ensure a good calibration, take pictures from different angles\n'
                      'Press "s" to save a frame.')
                take_mono_frames('camera1')
            print('\nA new window should have opened showing the checkerboard pattern with the detected corners.\n'
                  'If the detections are poor, skip the sample by pressing "s".\n'
                  'Otherwise, press any other key.\n'
                  'Anytime you want to close any emergent window, press ESC\n')
            K2, dist2, _, _ = get_intrinsics('./mono_frames/camera1*')
            print("Do you want to repeat the samples? (y/n)")
            print('(You should aim for a RMSE < 0.5)\n')
            ok = input('Answer: ')
            if ok == "N" or ok == "n":
                break
            elif ok not in ["n", "N", "y", "Y"]:
                print('Please, introduce a valid command:')
                ok = input('Answer: ')
        print('___' * 25, '\n')
        print("K matrix =\n", K2)
        print("\nDistortion coefficients \n", dist2)

        # Save parameters
        save_intrinsics('camera1', K2, dist2)
        print('\nIntrinsic parameters correctly saved for the second camera in "camera_parameters" folder.\n')
        print('Calibration for individual cameras completed!')
        print('===' * 25, '\n')

    # ___________________ STEREO CALIBRATION _______________________
    print('\n\033[4mSTEREO-CALIBRATION\033[0m')
    print('\nThe next step of the calibration process is to perform the stereo-calibration.\n\n'
          'Two pictures will be taken simultaneously with both cameras to estimate the '
          'parameters\nthat define the epipolar geometry of the system. '
          'Finally, the sample images\nwill be rectified, the disparity map calculated, and the '
          'parameters saved\non the "camera_parameters" folder.\n'
          '\nDo you want to take the pictures? (y/n)\n')
    ok = input('Answer: ')
    while ok not in ["n", "N", "y", "Y"]:
        print('Please, introduce a valid command:')
        ok = input('Answer: ')

    # Taking and saving stereo-frames
    if ok == "y" or ok == "Y":
        print('\nOpening both cameras...')
        print('This may take some time.\n'
              '\nPress "s" to save the frames')
        take_stereo_frames('camera0', 'camera1')

    # ___________________ IMPORT PARAMETERS ______________________
    # Read images
    img1 = cv.imread("./stereo_frames/camera0_1.png", 0)  # 0 imports the image in grayscale
    img2 = cv.imread("./stereo_frames/camera1_1.png", 0)

    # Import intrinsic parameters
    K1, dist1 = read_intrinsics('camera0')
    K2, dist2 = read_intrinsics('camera1')

    # ___________________ KEYPOINTS AND MATCH ______________________
    #
    list_kp1, list_kp2 = draw_keypoints_and_match(img1, img2)

    # ________________ FUNDAMENTAL AND ESSENTIAL MATRICES _________________
    #
    pts1 = np.int32(list_kp1)
    pts2 = np.int32(list_kp2)

    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
    E = calculate_E_matrix(F, K1, K2)

    # ________________ GET ROTATION AND TRANSLATION FROM E _________________
    # Define first camera as origin
    R1 = np.eye(3, dtype=np.float32)
    t1 = np.array([0., 0., 0.]).reshape((3, 1))

    camera_poses = extract_camera_poses(E)
    best_camera_pose = get_camera_pose(camera_poses, list_kp1)

    R2, t2 = best_camera_pose[0], best_camera_pose[1]
    t2 = np.array(t2).reshape(3, 1)

    # Save extrinsic parameters
    save_extrinsics('camera0', R1, t1)
    save_extrinsics('camera1', R2, t2)

    # ______________ COMPUTE EPILINES WITHOUT RECTIFICATION _________________
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    savename1 = os.path.join('images', 'not_rectified_1_with_epilines.png')
    cv.imwrite(savename1, img5)

    savename2 = os.path.join('images', 'not_rectified_2_with_epilines.png')
    cv.imwrite(savename2, img3)

    # _________________________ RECTIFICATION ____________________________

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))
    # Save rectification
    save_rectification(H1, H2)

    img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
    savename1 = os.path.join('images', 'rectified_1.png')
    cv.imwrite(savename1, img1_rectified)
    savename2 = os.path.join('images', 'rectified_2.png')
    cv.imwrite(savename2, img2_rectified)

    # Draw the rectified images
    fig1, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(img1_rectified, cmap="gray")
    axes[1].imshow(img2_rectified, cmap="gray")
    axes[0].axhline(250)
    axes[1].axhline(250)
    axes[0].axhline(450)
    axes[1].axhline(450)
    plt.suptitle("Rectified images")
    savename = os.path.join('images', 'rectified_images.png')
    plt.savefig(savename)
    plt.close()

    # _______________________ DISPARITY MAP _______________________
    # Using StereoBM
    # stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    # disparity = stereo.compute(img1_rectified, img2_rectified)

    # Using Stereo SGBM (better)
    block_size = 11
    min_disp = -128
    max_disp = 128

    num_disp = max_disp - min_disp
    uniquenessRatio = 5
    speckleWindowSize = 200

    speckleRange = 2
    disp12MaxDiff = 0

    stereo = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )
    disparity_SGBM = stereo.compute(img1_rectified, img2_rectified)

    # Normalize the values to a range from 0.255 for a grayscale image
    disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                  beta=0, norm_type=cv.NORM_MINMAX)
    disparity = np.uint8(disparity_SGBM)

    # Save depth map
    fig2, axes2 = plt.subplots(1, 1, figsize=(10, 8))
    axes2.imshow(disparity, cmap="gray")
    plt.suptitle("Depth Map")
    savename2 = os.path.join('images', 'depth_map.png')
    plt.savefig(savename2)
    plt.close()
