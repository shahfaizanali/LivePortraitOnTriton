
import numpy as np

try:
    import tritonclient.http as httpclient  # You can also use grpc if preferred
    from tritonclient.utils import triton_to_np_dtype
except ModuleNotFoundError:
    raise ImportError("Please install tritonclient: pip install tritonclient[all]")

# Mapping from NumPy dtypes to Triton dtypes
numpy_to_triton_dtype = {
    np.uint8: "UINT8",
    np.int8: "INT8",
    np.int16: "INT16",
    np.int32: "INT32",
    np.int64: "INT64",
    np.float16: "FP16",
    np.float32: "FP32",
    np.float64: "FP64",
    np.bool_: "BOOL",  # Updated here
}

class TritonPredictor:
    """
    Implements inference using Triton Inference Server.
    """

    def __init__(self, **kwargs):
        """
        :param model_name: The name of the model loaded in Triton.
        :param model_version: The version of the model to use. Optional.
        :param url: The Triton server URL, e.g., 'localhost:8000'.
        :param protocol: 'http' or 'grpc'.
        """
        self.model_name = kwargs.get("model_name", None)
        self.model_version = str(kwargs.get("model_version", "1"))
        self.url = kwargs.get("url", "localhost:8000")
        self.protocol = kwargs.get("protocol", "http")
        self.debug = kwargs.get("debug", False)

        if not self.model_name:
            raise ValueError("model_name must be provided")

        if self.protocol == "http":
            self.client = httpclient.InferenceServerClient(url=self.url, verbose=self.debug)
        else:
            raise NotImplementedError("Only HTTP protocol is supported in this implementation.")

        if not self.client.is_server_ready():
            raise RuntimeError("Triton server is not ready")

        if not self.client.is_model_ready(self.model_name, self.model_version):
            raise RuntimeError(f"Model {self.model_name} version {self.model_version} is not ready")

        # Retrieve model metadata
        self.model_metadata = self.client.get_model_metadata(self.model_name, self.model_version)
        self.model_config = self.client.get_model_config(self.model_name, self.model_version)

        # Access 'inputs' and 'outputs' as dictionary keys
        self.inputs = self.model_metadata.get('inputs', [])
        self.outputs = self.model_metadata.get('outputs', [])

        if self.debug:
            print(f"Triton Model Metadata: {self.model_metadata}")
            print(f"Triton Model Config: {self.model_config}")

    def input_spec(self):
        """
        Get the specs for the input tensors of the model.
        :return: A list of tuples containing (name, shape, datatype)
        """
        specs = []
        for inp in self.inputs:
            specs.append((inp['name'], inp['shape'], inp['datatype']))
            if self.debug:
                print(f"Triton input -> {inp['name']} -> {inp['shape']} -> {inp['datatype']}")
        return specs

    def output_spec(self):
        """
        Get the specs for the output tensors of the model.
        :return: A list of tuples containing (name, shape, datatype)
        """
        specs = []
        for out in self.outputs:
            specs.append((out['name'], out['shape'], out['datatype']))
            if self.debug:
                print(f"Triton output -> {out['name']} -> {out['shape']} -> {out['datatype']}")
        return specs

    def predict(self, feed_dict):
        """
        Execute inference on the provided inputs.
        :param feed_dict: A dictionary where keys are input tensor names and values are numpy arrays.
        :return: A dictionary with output tensor names as keys and numpy arrays as values.
        """
        inputs = []
        for name, data in feed_dict.items():
            if name not in [inp['name'] for inp in self.inputs]:
                raise ValueError(f"Input name {name} not found in model inputs")

            # Determine the Triton datatype
            dtype = numpy_to_triton_dtype.get(data.dtype.type, None)
            if dtype is None:
                raise ValueError(f"Unsupported data type: {data.dtype}")

            infer_input = httpclient.InferInput(name, data.shape, dtype)
            infer_input.set_data_from_numpy(data, binary_data=True)
            inputs.append(infer_input)

            if self.debug:
                print(f"Prepared input {name} with shape {data.shape} and dtype {dtype}")

        # Prepare outputs
        output_names = [out['name'] for out in self.outputs]
        outputs = [httpclient.InferRequestedOutput(name, binary_data=True) for name in output_names]

        # Perform inference
        results = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            model_version=self.model_version
        )

        # Extract output data
        output_data = {}
        for out in self.outputs:
            output_data[out['name']] = results.as_numpy(out['name'])
            if self.debug:
                print(f"Received output {out['name']} with shape {output_data[out['name']].shape} and dtype {output_data[out['name']].dtype}")

        return output_data

    def __del__(self):
        if hasattr(self, 'client'):
            del self.client

def get_predictor(**kwargs):
    """
    Factory method to get the Triton predictor.
    """
    return TritonPredictor(**kwargs)