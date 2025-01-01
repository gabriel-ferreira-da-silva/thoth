import numpy as np

class Initializer():
    def random(input_size, output_size):
        weights = np.random.rand(input_size, output_size) - 0.5
        bias = np.random.rand(1, output_size) - 0.5
        return weights, bias