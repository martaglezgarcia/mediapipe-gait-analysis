# This message includes the controllers set-points for the d2w subsystems:
# pelvis_ref_: Pelvis reference (set-point) respect to the d2w frame reference system - 3D position and 3D orientation.
# hip_ref_:    Hip reference (set-point) respect to the d2w frame reference system - 3D position.
# knee_ref_:   Knee reference (set-point) respect to the d2w frame reference system - 3D position.
# ankle_ref_:  Ankle reference (set-point) respect to the d2w frame reference system - 3D position.

std_msgs/Header header
std_msgs/Float32 r_gait_percentage
std_msgs/Float32 l_gait_percentage
geometry_msgs/Pose pelvis_ref
geometry_msgs/Pose r_hip_ref
geometry_msgs/Pose l_hip_ref
geometry_msgs/Pose r_knee_ref
geometry_msgs/Pose l_knee_ref
geometry_msgs/Pose r_ankle_ref
geometry_msgs/Pose l_ankle_ref
geometry_msgs/Twist vel_ref
