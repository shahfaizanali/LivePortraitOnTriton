import numpy as np
import tritonclient.grpc as grpcclient

# Only keeping what we need for Triton
class TritonPredictor:
    """
    TritonPredictor sends requests to a Triton Inference Server.
    You need a running Triton server with your model deployed.
    """

    def __init__(self, model_name, url="localhost:8001", model_version="", debug=False):
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self.debug = True

        # Initialize Triton gRPC client
        self.client = grpcclient.InferenceServerClient(url=self.url, verbose=self.debug)

        # Check if model is ready
        if not self.client.is_model_ready(self.model_name, self.model_version):
            raise RuntimeError(f"Model {self.model_name} not ready on Triton server at {self.url}")

        # Get model metadata
        self.model_metadata = self.client.get_model_metadata(self.model_name, self.model_version)
        self.model_config = self.client.get_model_config(self.model_name, self.model_version)

        # Parse input and output specs
        self.inputs = []
        for inp in self.model_metadata.inputs:
            self.inputs.append({"name": inp.name, "dtype": inp.datatype, "shape": inp.shape})

        self.outputs = []
        for out in self.model_metadata.outputs:
            self.outputs.append({"name": out.name, "dtype": out.datatype, "shape": out.shape})
            print(out.name, out.datatype, out.shape)

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

    def predict(self, feed_dict):
        # feed_dict: {input_name: np_array}
        triton_inputs = []
        for inp_meta in self.inputs:
            inp = grpcclient.InferInput(inp_meta["name"], feed_dict[inp_meta["name"]].shape, inp_meta["dtype"])
            inp.set_data_from_numpy(feed_dict[inp_meta["name"]])
            triton_inputs.append(inp)

        triton_outputs = []
        for out_meta in self.outputs:
            triton_outputs.append(grpcclient.InferRequestedOutput(out_meta["name"]))

        result = self.client.infer(self.model_name, triton_inputs, outputs=triton_outputs)

        out_dict = {}
        for out_meta in self.outputs:
            out_dict[out_meta["name"]] = result.as_numpy(out_meta["name"])
        return out_dict


def get_predictor(model_name, url="localhost:8001", debug=False):
    return TritonPredictor(model_name=model_name, url=url, debug=debug)
