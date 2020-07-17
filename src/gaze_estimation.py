import cv2
import numpy as np
import math
from base import Inference


# Reference: https://github.com/mdfazal/computer-pointer-controller-1/blob/master/gaze_estimation.py
class Gaze_Estimation(Inference):
    def __init__(self):
        super().__init__()

    def preprocess_input(self, left_eye_image, right_eye_image):
        re_resized = cv2.resize(right_eye_image, (self.get_gaze_input_shape()[3], self.get_gaze_input_shape()[2]))
        re_processed = np.transpose(np.expand_dims(re_resized, axis=0), (0, 3, 1, 2))

        le_resized = cv2.resize(left_eye_image, (self.get_gaze_input_shape()[3], self.get_gaze_input_shape()[2]))
        le_processed = np.transpose(np.expand_dims(le_resized, axis=0), (0, 3, 1, 2))

        return re_processed, le_processed

    def preprocess_output(self, outputs, head_pose_angle):
        gaze_vec = outputs[self.output_blob[0]][0]
        angle_r_fc = head_pose_angle[2]
        cosine = math.cos(angle_r_fc * math.pi / 180.0)
        sine = math.sin(angle_r_fc * math.pi / 180.0)

        x_val = gaze_vec[0] * cosine + gaze_vec[1] * sine
        y_val = -gaze_vec[0] * sine + gaze_vec[1] * cosine

        return (x_val, y_val), gaze_vec

    def predict(self, left_eye, right_eye, head_pose_angle):
        le_img_processed, re_img_processed = self.preprocess_input(left_eye.copy(), right_eye.copy())

        result = self.exec_network.infer({'head_pose_angles': head_pose_angle, 'left_eye_image': le_img_processed,
                                          'right_eye_image': re_img_processed})

        mouse_coords, gaze_vec = self.preprocess_output(result, head_pose_angle)

        return mouse_coords, gaze_vec
