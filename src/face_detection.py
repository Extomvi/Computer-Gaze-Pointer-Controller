import numpy as np
import  cv2
from base import Inference


class FaceDetection(Inference):
    def __init__(self):
        super().__init__()

    def preprocess_output(self, result, prob_threshold, *args, **kwargs):
        result = result[self.output_blob][0][0]
        coords = [[obj[3], obj[4], obj[5], obj[6]] for obj in result if obj[2] >= prob_threshold]
        return coords

    def predict(self, image, prob_threshold):
        coords = self.prediction_helper(image, self.preprocess_output, prob_threshold)

        coords = coords[0]

        h, w = image.shape[0], image.shape[1]

        coords = (coords * np.array([w, h, w, h])).astype(np.int32)
        crop_img = image[coords[1]:coords[3], coords[0]:coords[2]]

        return crop_img, coords

    def clean(self):
        return super(FaceDetection, self).clean()
