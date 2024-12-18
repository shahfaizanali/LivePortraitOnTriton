# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: warping_spade_model.py

from .new_predictor import get_predictor
import numpy as np
from .new_base_model import BaseModel
import torch

class WarpingSpadeModel(BaseModel):
    """
    WarpingSpade Model using Triton for inference
    """

    def __init__(self, **kwargs):
        super(WarpingSpadeModel, self).__init__(**kwargs)
        self.predictor = get_predictor(model_name="warping_spade-fix")
        if self.predictor is not None:
            self.input_shapes = self.predictor.input_spec()
            self.output_shapes = self.predictor.output_spec()
            print("Model Inputs:", self.predictor.inputs)



    def input_process(self, *data):
        # Data order: feature_3d, kp_source, kp_driving
        # The original code order is feature_3d, kp_driving, kp_source - ensure correct order:
        feature_3d, kp_driving, kp_source = data
        return feature_3d, kp_driving, kp_source

    def output_process(self, *data):
        # data[0] is the model output (assume one output)
        # Convert from NumPy to torch tensor
        out = torch.from_numpy(data[0])  # on CPU
        # Permute from NCHW to NHWC and scale to 0-255
        out = out.permute(0, 2, 3, 1)
        out = torch.clip(out, 0, 1) * 255
        return out[0]

    def predict(self, *data):
        feature_3d, kp_driving, kp_source = self.input_process(*data)

        # Create feed_dict for Triton
        feed_dict = {}
        # Assuming predictor.inputs are in the same order as the model expects:
        # Adjust if the model input names differ or order is different.
        feed_dict[self.predictor.inputs[0]['name']] = feature_3d.astype(np.float32)
        feed_dict[self.predictor.inputs[1]['name']] = kp_driving.astype(np.float32)
        feed_dict[self.predictor.inputs[2]['name']] = kp_source.astype(np.float32)

        # Triton inference
        preds_dict = self.predictor.predict(feed_dict)

        # Gather outputs
        outs = []
        for out_meta in self.predictor.outputs:
            outs.append(preds_dict[out_meta["name"]])

        # Process outputs
        outputs = self.output_process(*outs)
        return outputs
