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
        # Ensure the order matches what the model expects.
        # Original code returned feature_3d, kp_driving, kp_source
        feature_3d, kp_source, kp_driving = data
        return feature_3d, kp_driving, kp_source

    def output_process(self, *data):
        # data[0] is assumed to be a NumPy array from Triton
        out = data[0]  # shape: NCHW, float in [0, 1]
        
        # Clip and scale to 0-255
        np.clip(out, 0, 1, out=out)
        out = (out * 255).astype(np.uint8)

        # Transpose from NCHW to NHWC
        out = out.transpose(0, 2, 3, 1)

        # Return the first image in the batch
        return out[0]

    def predict(self, *data):
        feature_3d, kp_driving, kp_source = self.input_process(*data)

        # Create feed_dict for Triton
        feed_dict = {}
        feed_dict[self.predictor.inputs[0]['name']] = feature_3d.astype(np.float32)
        feed_dict[self.predictor.inputs[1]['name']] = kp_driving.astype(np.float32)
        feed_dict[self.predictor.inputs[2]['name']] = kp_source.astype(np.float32)

        preds_dict = self.predictor.predict(feed_dict)

        # Gather outputs
        outs = []
        for out_meta in self.predictor.outputs:
            outs.append(preds_dict[out_meta["name"]])

        # Process outputs completely in NumPy
        outputs = self.output_process(*outs)
        return outputs
