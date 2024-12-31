# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo0611@gmail.com
# @Project : FasterLivePortrait
# @FileName: stitching_model.py

from .new_base_model import BaseModel
from .async_predictor import get_predictor
import numpy as np

class StitchingModel(BaseModel):
    """
    StitchingModel using Triton for inference
    """

    def __init__(self, **kwargs):
        super(StitchingModel, self).__init__(**kwargs)
        model_name = kwargs.get("model_name", "stitching")
        self.predictor = get_predictor(model_name=model_name)
        # if self.predictor is not None:
        #     self.input_shapes = self.predictor.input_spec()
        #     self.output_shapes = self.predictor.output_spec()
        
    async def initialize(self):
        await self.predictor.initialize()

    def input_process(self, *data):
        # Assuming data[0] is a numpy array input
        return data[0]

    def output_process(self, *data):
        # Assuming a single output from the model
        return data[0]

    async def predict(self, *data):
        inp = self.input_process(*data)

        # Create feed_dict for Triton
        feed_dict = {}
        inp_meta = self.predictor.inputs[0]  # Assuming single input
        feed_dict[inp_meta['name']] = inp.astype(np.float32)

        preds_dict = await self.predictor.predict(feed_dict)

        # Gather outputs
        outs = []
        for out_meta in self.predictor.outputs:
            outs.append(preds_dict[out_meta["name"]])

        outputs = self.output_process(*outs)
        return outputs
