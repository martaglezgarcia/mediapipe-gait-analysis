# mediapipe-gait-analysis: calibration branch

> This branch from the mediapipe-gait-analysis repository includes all the files needed to perform a stereocalibration of two cameras prior to the 3D reconstruction using OpenCV and a checkerboard pattern.
> 

<aside>
❗ This code follows the steps provided by https://github.com/TemugeB/python_stereo_camera_calibrate

</aside>

***

**Contents**:

- “cal.py” - contains all the functions needed
- “calibration_main.py” - calls out the functions needed from “cal.py” to perform the stereocalibration of two cameras
- “calibration_settings.yaml” - YAML file with data needed for the calibration process using a checkerboard

**Packages required:**

```
OpenCV
Numpy
YAML
```

## Procedure

### STEP 0 - Edit calibration_settings.yaml file

First of all, edit the “calibration_settings.yaml” file according to your setup configuration. This is a preview of the content of the file and a brief explanation of each term:

```yaml
---
 camera0: 0
 camera1: 2
 frame_width: 1280
 frame_height: 720
 view_resize: 2
 stereo_cal_frames: 10
 mono_cal_frames: 10
 square_size: 3 # square board size in cm
 board_rows: 7
 board_cols: 7
 timer: 50  # 5 seconds
```

- camera0 and camera1 - device ids
- frame_width and frame_height - frame size
- view_resize - frame resize factor
- stereo_cal_frames and mono_cal_frames - number of pictures that will be taken to perform the calibrations for both cameras at the same time and each of them individually, respectively
- square_size - size of the checkerboard squares in cm
- board_rows and board_cols - number of crosses in the checkerboard (i.e. for a 8x8 checkerboard we would write 7 rows and 7 columns)
- timer - time between the pictures taken

Once the file is edited, run the code “calibration_main.py“. It is divided in different steps that will be explained below.

### STEP 1 - Load calibration settings

The calibration_settings.yaml” that was previously edited will be loaded using the *load_calibration_settings* function. 

### STEP 2 - Take images from each individual camera

Using the function *take_mono_frames* for both camera0 and camera1, a video captured by each camera at a time will be displayed. 

When the user presses the SPACEBAR, 10 pictures will be taken with a time of 5 seconds between them to correctly place the checkerboard in different positions, so that the pattern is well seen by the camera.

The images will be saved in a folder named “single_frames” and each picture will be named “camera id_frame number.png” (e.g. camera0_1.jpg)

### STEP 3 - Get and save intrinsic parameters for individual cameras

The **intrinsic parameters** of any camera are related to its internal optical and geometric characteristics (image optical center, focal length, lens distortion, skew coefficient, etc.)

The camera intrinsic matrix K is composed of the focal lengths ($f_x$,  $f_y$) in pixel units, and the image optical center ($c_x$, $c_y$), which is usually close to the center of the image.

$$
K = 
\begin{bmatrix}
f_x & 0   & c_x \\
0   & f_y & c_y \\
0   & 0   & 1 
\end{bmatrix}
$$

In this step, the OpenCV function *drawChessboardCorners* is a corner detection algorithm that will identify the checkerboard pattern. For each image taken, the program will display it with the estimations performed. The user should press “s” to skip the image if the pattern is not well defined or press any other key to take into account the measurement of the image displayed. 

Once the user has gone through all the images stored where the checkerboard pattern was identified, the *calibrateCamera* OpenCV function will return the intrinsic matrix K and the distortion coefficients of the camera. 

The function *save_intrinsic* will save these parameters in a newly created folder “camera_parameters” under the names “camera id_intrinsics.dat” (e.g. camera0_intrinsics.dat)

The RMSE value will be displayed, a good calibration would correspond to a value of less than 0.3

### STEP 4 - Take images for both cameras at the same time

Show the checkerboard to both cameras and press the SPACEBAR. The procedure is the same as the one before for individual cameras, but this tame both cameras will capture an image at the same time. 

The resulting images will be saved on a new folder named “stereo_frames” 

### STEP 5 - Obtain camera extrinsic parameters: rotation and translation matrices

The **extrinsic parameters** define a relation between the three-dimensional position and orientation of camera coordinates system compared with the world coordinates.

This step uses the images obtained in the previous step. Following the same procedure as before, the user will press “s” to skip a pair of image measurements or any other key to take them into account. The RMSE will be displayed and a value less than 0.3 should be aimed, up to 0.5 can be acceptable. If the desired RMSE is not obtained, repeat the previous and this step until a good value is obtained. 

### STEP 6 - Stereo calibration of extrinsic parameters

In order to correctly triangulate a 3D point, a world space origin and orientation needs to be defined. For the sake of simplicity, the camera0 will be defined as the world coordinate origin, and thus its rotation matrix will be an identity matrix, and its translation matrix, a zeros vector. 

This last step saves these parameters on the “camera_parameters” in two new files: camera0_rot_trans.dat and camera1_rot_trans.dat.

![image](https://docs.opencv.org/4.x/pinhole_camera_model.png)

## Resources

[OpenCV: Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

[https://github.com/TemugeB/python_stereo_camera_calibrate](https://github.com/TemugeB/python_stereo_camera_calibrate)
