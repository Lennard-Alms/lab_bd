
import numpy as np

class DummyProcessor:
    def __init__(self):
        pass

    def get_output_shape(self):
        return (512,)

    def execute(self, items):
        return np.full((items.shape[0], 512), 1)
