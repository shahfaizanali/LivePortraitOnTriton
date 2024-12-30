# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: motion_extractor_model.py

import numpy as np
from .new_base_model import BaseModel
from .async_predictor import get_predictor

class AppearanceFeatureExtractorModel(BaseModel):
    """
    AppearanceFeatureExtractorModel using Triton for inference
    """

    def __init__(self, **kwargs):
        super(AppearanceFeatureExtractorModel, self).__init__(**kwargs)
        self.predictor = get_predictor(model_name="appearance_feature_extractor")
        # if self.predictor is not None:
        #     self.input_shapes = self.predictor.input_spec()
        #     self.output_shapes = self.predictor.output_spec()

    def input_process(self, *data):
        img = data[0].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        return img[None]  # Add batch dimension

    def output_process(self, *data):
        # Assuming a single output from the model
        return data[0]

    async def predict(self, *data):
        await self.predictor.initialize()
        inp = self.input_process(*data)
        # Create feed_dict for Triton
        feed_dict = {}
        inp_meta = self.predictor.inputs[0]
        feed_dict[inp_meta['name']] = inp.astype(np.float32)

        preds_dict = await self.predictor.predict(feed_dict)

        # Gather outputs
        outs = []
        for out_meta in self.predictor.outputs:
            outs.append(preds_dict[out_meta["name"]])

        outputs = self.output_process(*outs)
        return outputs
