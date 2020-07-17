from base import Inference


class Head_Pose_Estimation(Inference):
    def __init__(self):
        super().__init__()

    @staticmethod
    def preprocess_output(outputs, *args, **kwargs):
        hp_out = [outputs[key][0][0] for key in outputs]
        return [hp_out[2], hp_out[0], hp_out[1]]

    def predict(self, image):
        coords = self.prediction_helper(image, self.preprocess_output)
        return coords
