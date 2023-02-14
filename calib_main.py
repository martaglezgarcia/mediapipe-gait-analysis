import math
import cv2 as cv
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import yaml

from calib_tools import *


if __name__ == '__main__':

    # STEP 0 : LOAD CALIBRATION SETTINGS
    calibration_settings = load_calibration_settings('calibration_settings.yaml')

    #___________________ CALIBRATION_______________________
    # STEP 1: GET CALIBRATION FRAMES FOR INDIVIDUAL CAMERAS
    #take_mono_frames('camera0')
    #take_mono_frames('camera1')

    # STEP 2: ESTIMATE AND SAVE INTRINSIC PARAMETERS FOR BOTH CAMERAS
    K1, _, _, _ = get_intrinsics('./single_frames/camera0*')
    print('Intrinsic parameters')
    print("K matrix for camera0", K1)
    print("==" * 20, '\n')

    K2, _, _, _ = get_intrinsics('./single_frames/camera0*')
    print("K matrix for camera1", K2)
    print("==" * 20, '\n')

    dist1 = np.array([0., 0., 0., 0., 0.]).reshape((1, 5))
    dist2 = np.array([0., 0., 0., 0., 0.]).reshape((1, 5))

    save_intrinsics('camera0', K1, dist1)
    save_intrinsics('camera1', K2, dist2)
    print('Intrinsic parameters correctly saved!')
    print('Calibration for individual cameras completed.')

    # STEP 3: TAKE 1 STEREO-FRAME
    #take_stereo_frames('camera0', 'camera1')

    # STEP 4: CALCULATE FUNDAMENTAL AND ESSENTIAL MATRICES

    # get images
    img1 = cv.imread("./stereo_frames/camera0_1.png", 0)
    img2 = cv.imread("./stereo_frames/camera1_1.png", 0)

    # lo que hace que le cueste es el *1 en vez de *0.3
    # con 0.5 va bien pero no más
    width = int(img1.shape[1] * 0.5)
    height = int(img1.shape[0] * 0.5)

    # make images same size
    img1 = cv.resize(img1, (width, height), interpolation=cv.INTER_AREA)
    img2 = cv.resize(img2, (width, height), interpolation=cv.INTER_AREA)

    while 1:
        try:
            list_kp1, list_kp2 = draw_keypoints_and_match(img1, img2)

            # Calculation of F and E using 8-point algorithm and RANSAC
            F = RANSAC_F_matrix([list_kp1, list_kp2])
            print('Fundamental and Essential matrices')
            print("==" * 20, '\n')
            print("F matrix", F)
            print()
            E = calculate_E_matrix(F, K1, K2)
            print("E matrix", E)
            print("==" * 20, '\n')
            # Get R and t from E
            camera_poses = extract_cameraposes(E)
            best_camera_pose = get_camerapose(camera_poses, list_kp1)
            R = best_camera_pose[0]
            t = best_camera_pose[1]
            print("Best Camera Pose:")
            print("==" * 20)
            print("Rotation", R)
            print()
            print("Translation", t)
            print("==" * 20, '\n')

            pts1 = np.int32(list_kp1)
            pts2 = np.int32(list_kp2)

            # ____________________________RECTIFICATION________________________________
            rectified_pts1, rectified_pts2, img1_rectified, img2_rectified, H1, H2 = rectification(img1, img2, pts1,
                                                                                                   pts2, F)
            print('Rectified images correctly saved!')
            break
        except Exception as e:
            continue

    # Save extrinsic parameters. Camera0 is the world-coordinate origin
    R0 = np.eye(3, dtype=np.float32)
    t0 = np.array([0., 0., 0.]).reshape((3, 1))

    t = np.array(t).reshape(3, 1)
    save_extrinsics('camera0', R0, t0)
    save_extrinsics('camera1', R, t)
    save_rectification(H1, H2)

    # STEP 5: COMPUTE EPILINES AND SAVE IMAGES
    # Find epilines corresponding to points in right image (second image) and drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(rectified_pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1_rectified, img2_rectified, lines1, rectified_pts1, rectified_pts2)

    # Find epilines corresponding to points in left image (first image) and drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(rectified_pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2_rectified, img1_rectified, lines2, rectified_pts2, rectified_pts1)

    # create folder if it does not exist
    if not os.path.exists('images'):
        os.mkdir('images')

    savename1 = os.path.join('images', 'rectified_1_with_epilines.png')
    cv.imwrite(savename1, img5)

    savename2 = os.path.join('images', 'rectified_2_with_epilines.png')
    cv.imwrite(savename2, img3)

    '''
    # STEP 6: DISPARITY MAP
    disparity_map_unscaled, disparity_map_scaled = get_disparity(img1_rectified, img2_rectified)

    plt.figure(1)
    plt.title('Disparity Map Grayscale')
    plt.imshow(disparity_map_scaled, cmap='gray')
    plt.savefig('./images/disparity_map.png')
    print('Disparity map correctly saved!')
    '''