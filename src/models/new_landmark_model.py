# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: landmark_model.py

from .new_base_model import BaseModel
from .new_predictor import get_predictor
import cv2
import numpy as np
from src.utils.crop import crop_image, _transform_pts


class LandmarkModel(BaseModel):
    """
    Landmark Model using Triton for inference
    """

    def __init__(self, **kwargs):
        super(LandmarkModel, self).__init__(**kwargs)
        self.dsize = 224
        self.predictor = get_predictor(model_name="landmark")
        if self.predictor is not None:
            self.input_shapes = self.predictor.input_spec()
            self.output_shapes = self.predictor.output_spec()
        # Assume self.predictor is already a TritonPredictor instance
        # configured in the parent BaseModel or initialization code.

    def input_process(self, *data):
        if len(data) > 1:
            img_rgb, lmk = data
        else:
            img_rgb = data[0]
            lmk = None

        if lmk is not None:
            crop_dct = crop_image(img_rgb, lmk, dsize=self.dsize, scale=1.5, vy_ratio=-0.1)
            img_crop_rgb = crop_dct['img_crop']
        else:
            # Fallback: resize directly (not recommended, but kept from original code)
            img_crop_rgb = cv2.resize(img_rgb, (self.dsize, self.dsize))
            scale = max(img_rgb.shape[:2]) / self.dsize
            crop_dct = {
                'M_c2o': np.array([
                    [scale, 0., 0.],
                    [0., scale, 0.],
                    [0., 0., 1.],
                ], dtype=np.float32),
            }

        # Preprocess to [1,3,H,W], normalized
        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]
        return inp, crop_dct

    def output_process(self, preds, crop_dct):
        # preds is a list of outputs; we assume preds[0] is the landmark output
        # adjust as per your model's specific output indexing if needed
        for i, output in enumerate(preds):
          print(f"Shape of output {i}: {output.shape}")
        lmk = preds[2].reshape(-1, 2) * self.dsize
        lmk = _transform_pts(lmk, M=crop_dct['M_c2o'])
        return lmk

    def predict(self, *data):
        # Prepare input
        inp, crop_dct = self.input_process(*data)

        # Create feed_dict for Triton
        feed_dict = {}
        # Assuming the model has one input
        inp_meta = self.predictor.inputs[0]
        feed_dict[inp_meta['name']] = inp.astype(np.float32)

        # Inference call to Triton
        preds_dict = self.predictor.predict(feed_dict)

        # Gather outputs in the order they appear in self.predictor.outputs
        outs = []
        for out_meta in self.predictor.outputs:
            outs.append(preds_dict[out_meta["name"]])

        # Process outputs into desired format
        outputs = self.output_process(outs, crop_dct)
        return outputs
