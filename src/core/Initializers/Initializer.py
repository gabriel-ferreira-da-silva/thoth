import numpy as np

class Initializer():
    @staticmethod
    def random(input_size, output_size):
        weights = np.random.rand(input_size, output_size) - 0.5
        bias = np.random.rand(1, output_size) - 0.5
        return weights, bias
    

    @staticmethod
    def cosine(input_size, output_size):     
        weights = np.zeros((input_size, output_size))
        for i in range(input_size):
            for j in range(output_size):
                weights[i, j] = np.cos( i + j) 

        bias = np.zeros((1, output_size))
        for j in range(output_size):
            bias[0, j] = np.cos(j) 
        return weights, bias

    @staticmethod
    def uniform(input_size, output_size, value=0.001):
        weights = np.full(( input_size, output_size), value)
        bias = np.full((1, output_size), value)
        return weights, bias 