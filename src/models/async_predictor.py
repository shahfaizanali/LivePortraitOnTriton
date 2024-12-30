import numpy as np
import tritonclient.http.aio as httpclient

class TritonPredictor:
    """
    TritonPredictor sends requests to a Triton Inference Server using the asynchronous HTTP client.
    """

    def __init__(self, model_name, url="localhost:8000", model_version="", debug=False):
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self.debug = debug
        self.client = httpclient.InferenceServerClient(url=self.url, verbose=self.debug)

        self.inputs = []
        self.outputs = []
        self.model_metadata = None
        self.model_config = None
        self.initialized = False

    async def initialize(self):
        if not self.initialized:
            self.initialized = True
            # Check if model is ready
            if not await self.client.is_model_ready(self.model_name, self.model_version):
                raise RuntimeError(f"Model {self.model_name} not ready on Triton server at {self.url}")

            # Get model metadata and config asynchronously
            self.model_metadata = await self.client.get_model_metadata(self.model_name, self.model_version)
            self.model_config = await self.client.get_model_config(self.model_name, self.model_version)

            # Parse input and output specs
            for inp in self.model_metadata['inputs']:
                self.inputs.append({"name": inp['name'], "dtype": inp['datatype'], "shape": inp['shape']})

            for out in self.model_metadata['outputs']:
                self.outputs.append({"name": out['name'], "dtype": out['datatype'], "shape": out['shape']})
       

    def input_spec(self):
        specs = []
        for o in self.inputs:
            specs.append((o["name"], o["shape"], o["dtype"]))
            if self.debug:
                print(f"Triton input -> {o['name']} -> {o['shape']}, dtype={o['dtype']}")
        return specs

    def output_spec(self):
        specs = []
        for o in self.outputs:
            specs.append((o["name"], o["shape"], o["dtype"]))
            if self.debug:
                print(f"Triton output -> {o['name']} -> {o['shape']}, dtype={o['dtype']}")
        return specs

    async def predict(self, feed_dict):
        # if not self.initialized:
        #     await self.initialize()
        #     self.initialized = True
        # feed_dict: {input_name: np_array}
        triton_inputs = []
        for inp_meta in self.inputs:
            inp_data = feed_dict[inp_meta["name"]]
            inp = httpclient.InferInput(inp_meta["name"], inp_data.shape, inp_meta["dtype"])
            inp.set_data_from_numpy(inp_data)
            triton_inputs.append(inp)

        triton_outputs = []
        for out_meta in self.outputs:
            triton_outputs.append(httpclient.InferRequestedOutput(out_meta["name"]))

        # Send inference request asynchronously
        result = await self.client.infer(
            self.model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
            model_version=self.model_version
        )

        out_dict = {}
        for out_meta in self.outputs:
            out_dict[out_meta["name"]] = result.as_numpy(out_meta["name"])
        return out_dict


def get_predictor(model_name, url="localhost:8000", debug=False):
  return TritonPredictor(model_name=model_name, url=url, debug=debug)
