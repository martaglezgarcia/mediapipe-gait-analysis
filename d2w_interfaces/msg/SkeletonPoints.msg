# This message includes the skeleton keypoints:
#
# shoulder_ref_ : shoulder landmarks detected by MediaPipe Pose
# elbow_ref_    : elbow landmarks detected by MediaPipe Pose
# wrist_ref_    : wrist landmarks detected by MediaPipe Pose
# pinky_ref_    : pinky landmarks detected by MediaPipe Pose
# index_ref_    : index landmarks detected by MediaPipe Pose
# thumb_ref     : thumb landmarks detected by MediaPipe Pose
# hip_ref_      : hip landmarks detected by MediaPipe Pose
# knee_ref_     : knee landmarks detected by MediaPipe Pose
# ankle_ref_    : ankle landmarks detected by MediaPipe Pose
# heel_ref_     : heel landmarks detected by MediaPipe Pose
# foot_ref_     : foot index landmarks detected by MediaPipe Pose

std_msgs/Header header
# upper body
geometry_msgs/Pose r_shoulder_ref
geometry_msgs/Pose l_shoulder_ref
geometry_msgs/Pose r_elbow_ref
geometry_msgs/Pose l_elbow_ref
# hands
geometry_msgs/Pose r_wrist_ref
geometry_msgs/Pose l_wrist_ref
geometry_msgs/Pose r_pinky_ref
geometry_msgs/Pose l_pinky_ref
geometry_msgs/Pose r_index_ref
geometry_msgs/Pose l_index_ref
geometry_msgs/Pose r_thumb_ref
geometry_msgs/Pose l_thumb_ref
# lower body
geometry_msgs/Pose r_hip_ref
geometry_msgs/Pose l_hip_ref
geometry_msgs/Pose r_knee_ref
geometry_msgs/Pose l_knee_ref
geometry_msgs/Pose r_ankle_ref
geometry_msgs/Pose l_ankle_ref
geometry_msgs/Pose r_heel_ref
geometry_msgs/Pose l_heel_ref
geometry_msgs/Pose r_foot_ref
geometry_msgs/Pose l_foot_ref