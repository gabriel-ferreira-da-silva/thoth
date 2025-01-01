from .BaseLayer import BaseLayer as Layer
from ..Initializers.Initializer import Initializer
import numpy as np


class FullyActiveConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights, self.bias = Initializer.random(input_size, output_size) 
    
    def forward_propagation(self, input, activation):
        self.input = input
        self.linear_output = np.dot(self.input, self.weights) + self.bias
        self.output = activation(self.linear_output)
        return self.output
    
    def backward_propagation(self, output_error, activation_prime, learning_rate):
        
        output_error_active = activation_prime(self.linear_output) * output_error
        
        input_error = np.dot(output_error_active, self.weights.T)
        weights_error = np.dot(self.input.T, output_error_active)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * np.sum(output_error_active, axis=0, keepdims=True)

        return input_error