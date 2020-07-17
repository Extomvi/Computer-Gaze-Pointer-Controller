import os
import sys
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore


class Inference:
    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU", cpu_extension=None, num_requests=0):
        """
       Load the model given IR files.
       Defaults to CPU as device for use in the workspace.
       Synchronous requests made within.
       """
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
            log.info("CPU extension loaded: {}".format(cpu_extension))

        # Read the IR as a IENetwork
        try:
            self.network = IENetwork(model=model_xml, weights=model_bin)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you entered the correct model path?")

        # Check Network layer support
        if "CPU" in device:
            supported_layers = self.plugin.query_network(self.network, "CPU")
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(device, ', '.join(not_supported_layers)))
                log.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
                sys.exit(1)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device, num_requests=num_requests)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return

    def preprocess_input(self, image):
        n, c, h, w = self.get_input_shape()
        img = cv2.resize(image, (w, h))
        img = img.transpose((2, 0, 1))
        img = img.reshape((n, c, h, w))

        return img

    def get_input_shape(self):
        """
        Gets the input shape of the network
        """
        return self.network.inputs[self.input_blob].shape

    def get_gaze_input_shape(self):
        # Get the input layer
        self.input_blob = [i for i in self.network.inputs.keys()]
        self.output_blob = [i for i in self.network.outputs.keys()]

        return self.network.inputs[self.input_blob[1]].shape

    def prediction_helper(self, image, preprocess_output, prob_threshold=0.6):
        img_processed = self.preprocess_input(image.copy())
        result = self.exec_network.infer(inputs={self.input_blob: img_processed})
        coords = preprocess_output(result, prob_threshold)

        return coords

    # def clean(self):
    #     """
    #     Deletes all the instances
    #     :return: None
    #     """
    #     print('SEE ME HERE')
    #     del self.exec_network
    #     del self.plugin
    #     del self.network
