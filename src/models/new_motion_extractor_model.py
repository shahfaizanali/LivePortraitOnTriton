# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: motion_extractor_model.py

import numpy as np
from .new_base_model import BaseModel
from .async_predictor import get_predictor

def headpose_pred_to_degree(pred):
    if pred.ndim > 1 and pred.shape[1] == 66:
        idx_array = np.arange(0, 66)
        pred = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, pred)
        degree = np.sum(pred * idx_array, axis=1) * 3 - 97.5
        return degree
    return pred

class MotionExtractorModel(BaseModel):
    """
    MotionExtractorModel using Triton for inference
    """

    def __init__(self, **kwargs):
        super(MotionExtractorModel, self).__init__(**kwargs)
        self.flag_refine_info = kwargs.get("flag_refine_info", True)
        self.predictor = get_predictor(model_name="motion_extractor")
        # if self.predictor is not None:
        #     self.input_shapes = self.predictor.input_spec()
        #     self.output_shapes = self.predictor.output_spec()

    async def initialize(self):
        await self.predictor.initialize()

    def input_process(self, *data):
        img = data[0].astype(np.float32)
        img /= 255.0
        img = np.transpose(img, (2, 0, 1))
        return img[None]

    def output_process(self, *data):
        # Data order: kp, pitch, yaw, roll, t, exp, scale
        kp, pitch, yaw, roll, t, exp, scale = data
        if self.flag_refine_info:
            bs = kp.shape[0]
            pitch = headpose_pred_to_degree(pitch)[:, None]  # Bx1
            yaw = headpose_pred_to_degree(yaw)[:, None]      # Bx1
            roll = headpose_pred_to_degree(roll)[:, None]    # Bx1
            kp = kp.reshape(bs, -1, 3)    # BxNx3
            exp = exp.reshape(bs, -1, 3)  # BxNx3
        return pitch, yaw, roll, t, exp, scale, kp

    async def predict(self, *data):
        # Preprocess input
        inp = self.input_process(*data)

        # Create feed_dict for Triton
        feed_dict = {}
        inp_meta = self.predictor.inputs[0]
        feed_dict[inp_meta['name']] = inp.astype(np.float32)

        # Inference call to Triton
        preds_dict = await self.predictor.predict(feed_dict)

        # Gather outputs
        outs = []
        for out_meta in self.predictor.outputs:
            outs.append(preds_dict[out_meta["name"]])

        # Process outputs
        outputs = self.output_process(*outs)
        return outputs
