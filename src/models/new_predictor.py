import numpy as np
import torch
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

class TritonTensorRTPredictor:
    """
    Implements inference for TensorRT models served via Triton Inference Server
    Compatible with existing predictor interface
    """

    def __init__(self, **kwargs):
        """
        Initialize Triton client connection
        
        :param model_path: Path to the model (used for identification)
        :param url: Triton server URL (default: localhost:8000)
        :param model_name: Name of the model in Triton (optional)
        :param predict_type: Prediction type (default: 'triton')
        :param debug: Enable debug logging
        """
        self.model_path = kwargs.get('model_path', '')
        self.url = kwargs.get('url', 'localhost:8000')
        self.model_name = kwargs.get('model_name', '')
        self.debug = kwargs.get('debug', False)
        
        # If model_name not provided, use last part of model_path as default
        if not self.model_name and self.model_path:
            self.model_name = self.model_path.split('/')[-1].split('.')[0]
        
        assert self.model_name, "Model name must be provided!"
        
        try:
            self.client = httpclient.InferenceServerClient(url=self.url)
            
            # Fetch model metadata
            self.model_metadata = self.client.get_model_metadata(
                model_name=self.model_name
            )
            
            # Parse input and output specifications
            self._parse_io_specs()
        
        except InferenceServerException as e:
            raise RuntimeError(f"Failed to connect to Triton server: {e}")

    def _parse_io_specs(self):
        """
        Parse input and output specifications from model metadata
        """
        self.inputs = []
        self.outputs = []
        
        for input_meta in self.model_metadata['inputs']:
            input_spec = {
                'name': input_meta['name'],
                'shape': input_meta['shape'],
                'dtype': input_meta['datatype']
            }
            self.inputs.append(input_spec)
            
            if self.debug:
                print(f"Triton input: {input_spec}")
        
        for output_meta in self.model_metadata['outputs']:
            output_spec = {
                'name': output_meta['name'],
                'shape': output_meta['shape'],
                'dtype': output_meta['datatype']
            }
            self.outputs.append(output_spec)
            
            if self.debug:
                print(f"Triton output: {output_spec}")

    def input_spec(self):
        """
        Get the specs for the input tensors of the network
        :return: List of input specifications
        """
        return [(i['name'], i['shape'], i['dtype']) for i in self.inputs]

    def output_spec(self):
        """
        Get the specs for the output tensors of the network
        :return: List of output specifications
        """
        return [(o['name'], o['shape'], o['dtype']) for o in self.outputs]

    def predict(self, *data, **kwargs):
        """
        Execute inference on Triton server
        Supports both TensorRT-style input (feed_dict, stream) and ORT-style direct input
        
        :param data: Input tensor(s)
        :param kwargs: Additional arguments (stream for TensorRT-style)
        :return: Inference results
        """
        try:
            # Prepare inputs for Triton
            inputs = []
            
            # Handle different input styles
            if len(data) == 1 and isinstance(data[0], dict):
                # TensorRT-style feed_dict with stream
                feed_dict = data[0]
                stream = kwargs.get('stream', None)
                
                for input_spec in self.inputs:
                    input_name = input_spec['name']
                    input_tensor = feed_dict[input_name]
                    
                    # Convert torch tensor to numpy if necessary
                    if torch.is_tensor(input_tensor):
                        input_tensor = input_tensor.cpu().numpy()
                    
                    # Create Triton InferInput
                    triton_input = httpclient.InferInput(
                        input_name, 
                        input_tensor.shape, 
                        input_spec['dtype']
                    )
                    triton_input.set_data_from_numpy(input_tensor)
                    inputs.append(triton_input)
            else:
                # ORT-style direct input
                for i, input_tensor in enumerate(data):
                    input_spec = self.inputs[i]
                    
                    # Convert torch tensor to numpy if necessary
                    if torch.is_tensor(input_tensor):
                        input_tensor = input_tensor.cpu().numpy()
                    
                    # Create Triton InferInput
                    triton_input = httpclient.InferInput(
                        input_spec['name'], 
                        input_tensor.shape, 
                        input_spec['dtype']
                    )
                    triton_input.set_data_from_numpy(input_tensor)
                    inputs.append(triton_input)
            
            # Prepare outputs
            outputs = []
            for output_spec in self.outputs:
                outputs.append(
                    httpclient.InferRequestedOutput(output_spec['name'])
                )
            
            # Send inference request
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            # Process and return results
            results = []
            for output_spec in self.outputs:
                output_name = output_spec['name']
                output_tensor = response.as_numpy(output_name)
                results.append(output_tensor)
            
            # Return a single tensor if only one output
            return results[0] if len(results) == 1 else results
        
        except InferenceServerException as e:
            raise RuntimeError(f"Inference failed: {e}")

    def __del__(self):
        """
        Cleanup Triton client
        """
        if hasattr(self, 'client'):
            del self.client


numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

def get_predictor(**kwargs):
  return TritonTensorRTPredictor(**kwargs)
    