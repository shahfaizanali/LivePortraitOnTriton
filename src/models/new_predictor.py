import numpy as np
import torch
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

import numpy as np
import torch
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

# Mapping for Triton dtype strings to NumPy/Torch dtypes
TRITON_DTYPE_MAP = {
    'FP32': np.float32,
    'FP16': np.float16,
    'INT32': np.int32,
    'INT64': np.int64,
    'UINT8': np.uint8,
    'BOOL': np.bool_
}

class TritonTensorRTPredictor:
    """
    Implements inference for TensorRT models served via Triton Inference Server
    Enhanced input handling and dtype conversion
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
        Converts Triton dtype to NumPy dtype
        """
        self.inputs = []
        self.outputs = []
        
        for input_meta in self.model_metadata['inputs']:
            # Convert Triton dtype to NumPy dtype
            try:
                dtype = TRITON_DTYPE_MAP.get(input_meta['datatype'], input_meta['datatype'])
            except Exception:
                dtype = input_meta['datatype']
            
            input_spec = {
                'name': input_meta['name'],
                'shape': input_meta['shape'],
                'dtype': dtype
            }
            self.inputs.append(input_spec)
            
            if self.debug:
                print(f"Triton input: {input_spec}")
        
        for output_meta in self.model_metadata['outputs']:
            # Convert Triton dtype to NumPy dtype
            try:
                dtype = TRITON_DTYPE_MAP.get(output_meta['datatype'], output_meta['datatype'])
            except Exception:
                dtype = output_meta['datatype']
            
            output_spec = {
                'name': output_meta['name'],
                'shape': output_meta['shape'],
                'dtype': dtype
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
        Supports multiple input styles
        
        :param data: Input tensor(s)
        :param kwargs: Additional arguments (stream for TensorRT-style)
        :return: Inference results
        """
        try:
            # Prepare inputs for Triton
            inputs = []
            
            # Handle different input styles
            if len(data) == 1:
                if isinstance(data[0], dict):
                    # TensorRT-style feed_dict
                    feed_dict = data[0]
                    stream = kwargs.get('stream', None)
                    
                    for input_spec in self.inputs:
                        input_name = input_spec['name']
                        input_tensor = feed_dict[input_name]
                        
                        # Handle different input types
                        if isinstance(input_tensor, dict):
                            # If input is a dict, try to extract the tensor
                            input_tensor = input_tensor.get('tensor', input_tensor)
                        
                        # Convert torch tensor to numpy if necessary
                        if torch.is_tensor(input_tensor):
                            input_tensor = input_tensor.cpu().numpy()
                        
                        # Ensure correct dtype
                        input_tensor = np.array(input_tensor, dtype=input_spec['dtype'])
                        
                        # Create Triton InferInput
                        triton_input = httpclient.InferInput(
                            input_name, 
                            input_tensor.shape, 
                            # Convert back to Triton dtype string for compatibility
                            next(k for k, v in TRITON_DTYPE_MAP.items() if v == input_spec['dtype'])
                        )
                        triton_input.set_data_from_numpy(input_tensor)
                        inputs.append(triton_input)
                else:
                    # Direct input tensor
                    input_tensor = data[0]
                    
                    # Handle different input types
                    if isinstance(input_tensor, dict):
                        # If input is a dict, try to extract the tensor
                        input_tensor = input_tensor.get('tensor', input_tensor)
                    
                    # Convert torch tensor to numpy if necessary
                    if torch.is_tensor(input_tensor):
                        input_tensor = input_tensor.cpu().numpy()
                    
                    input_spec = self.inputs[0]
                    
                    # Ensure correct dtype
                    input_tensor = np.array(input_tensor, dtype=input_spec['dtype'])
                    
                    # Create Triton InferInput
                    triton_input = httpclient.InferInput(
                        input_spec['name'], 
                        input_tensor.shape, 
                        # Convert back to Triton dtype string for compatibility
                        next(k for k, v in TRITON_DTYPE_MAP.items() if v == input_spec['dtype'])
                    )
                    triton_input.set_data_from_numpy(input_tensor)
                    inputs.append(triton_input)
            else:
                # Multiple direct inputs
                for i, input_tensor in enumerate(data):
                    input_spec = self.inputs[i]
                    
                    # Handle different input types
                    if isinstance(input_tensor, dict):
                        # If input is a dict, try to extract the tensor
                        input_tensor = input_tensor.get('tensor', input_tensor)
                    
                    # Convert torch tensor to numpy if necessary
                    if torch.is_tensor(input_tensor):
                        input_tensor = input_tensor.cpu().numpy()
                    
                    # Ensure correct dtype
                    input_tensor = np.array(input_tensor, dtype=input_spec['dtype'])
                    
                    # Create Triton InferInput
                    triton_input = httpclient.InferInput(
                        input_spec['name'], 
                        input_tensor.shape, 
                        # Convert back to Triton dtype string for compatibility
                        next(k for k, v in TRITON_DTYPE_MAP.items() if v == input_spec['dtype'])
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
            results = {}
            for output_spec in self.outputs:
                output_name = output_spec['name']
                output_tensor = response.as_numpy(output_name)
                
                # Ensure correct dtype
                output_tensor = output_tensor.astype(output_spec['dtype'])
                
                results[output_name] = torch.from_numpy(output_tensor)
            
            return results
        
        except InferenceServerException as e:
            raise RuntimeError(f"Inference failed: {e}")

    def __del__(self):
        """
        Cleanup Triton client
        """
        if hasattr(self, 'client'):
            del self.client

def get_predictor(**kwargs):
  return TritonTensorRTPredictor(**kwargs)
    