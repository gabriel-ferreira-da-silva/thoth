from .BaseLayer import BaseLayer as Layer
from ..Initializers.Initializer import Initializer
import numpy as np


class FullyActiveConnectedLayer(Layer):
    def __init__(self, input_size, output_size, initializer="random"):
        self.initializer = initializer
        self.weights =None
        self.bias = None
        self.weightsGradient=None
        self.biasGradient=None
        self.set(input_size, output_size)
    
    def getWeights(self):
        return self.weights
    
    def getBias(self):
        return self.bias
    
    def getWeightsGradient(self):
        return self.weightsGradient
    
    def getBiasGradient(self):
        return self.biasGradient
    
    def set(self, input_size, output_size):
        init = None
        
        if self.initializer=="uniform":
            init = Initializer.uniform
        elif self.initializer=="random":
            init = Initializer.random
        elif self.initializer=="cosine":
            init = Initializer.cosine
        else:
            init = Initializer.random

        self.weights, self.bias = init(input_size, output_size) 

    
    def forward_propagation(self, input, activation):
    
        self.input = input
        self.linear_output = np.dot(self.input, self.weights) + self.bias
        self.output = activation(self.linear_output)
        return self.output
    
    def backward_propagation(self, output_error, activation_prime, learning_rate):
    
        output_error_active = activation_prime(self.linear_output) * output_error
        
        input_error = np.dot(output_error_active, self.weights.T)
        weights_error = np.dot(self.input.T, output_error_active)

        self.weightsGradient = weights_error
        self.biasGradient = np.sum(output_error_active, axis=0, keepdims=True)

        self.weights -= learning_rate * self.weightsGradient
        self.bias -= learning_rate * self.biasGradient

        return input_error