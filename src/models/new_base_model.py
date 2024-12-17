import copy
from .new_predictor import get_predictor

class BaseModel:
    """
    Base class for models using Triton inference.
    """

    def __init__(self, **kwargs):
        self.kwargs = copy.deepcopy(kwargs)

    def input_process(self, *data):
        """
        Input pre-processing (should be implemented in subclasses).
        """
        pass

    def output_process(self, *data):
        """
        Output post-processing (should be implemented in subclasses).
        """
        pass

    def predict(self, *data):
        """
        Prediction method (should be implemented in subclasses).
        """
        pass

    def __del__(self):
        """
        Clean up predictor instance.
        """
        if self.predictor is not None:
            del self.predictor
