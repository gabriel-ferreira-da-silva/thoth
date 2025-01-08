from .BaseLayer import BaseLayer as Layer
from ..Initializers.Initializer import Initializer
from ..Initializers.InitializerSelector import InitializerSelector
from ..optimizers.OptimizerSelector import OptimizerSelector
from ..optimizers.Optimizers import Optimizers
import numpy as np


class FullyActiveConnectedLayer(Layer):
    def __init__(self, input_size, output_size, initializer="random", optimizer="momentum"):
        self.weights =None
        self.bias = None
        self.weightsGradient=None
        self.biasGradient=None
        self.optimizer = None
        self.initializer = None
        self.optimizer_name =None
        self.initializer_name =None
        self.setOptimizer(optimizer)
        self.setInitializer(initializer)
        self.set(input_size, output_size)
    
    def getWeights(self):
        return self.weights
    
    def getInitializerName(self):
        return self.initializer_name
    
    def getOptimizerName(self):
        return self.optimizer_name
    
    def getBias(self):
        return self.bias
    
    def getWeightsGradient(self):
        return self.weightsGradient
    
    def getBiasGradient(self):
        return self.biasGradient
    
    def setOptimizer(self, optimizer):

        if optimizer not in OptimizerSelector:
            optimizer = "momentum"
        
        self.optimizer = OptimizerSelector[optimizer]()
        self.optimizer.initialize(self.weights, self.bias)
        self.optimizer_name=optimizer
    
    def setInitializer(self, initializer):
        if initializer not in InitializerSelector:
            initializer = "random"
        
        self.initializer = InitializerSelector[initializer]
        self.initializer_name = initializer
    
    def set(self, input_size, output_size):
    
        self.weights, self.bias = self.initializer(input_size, output_size) 
        self.optimizer.initialize(self.weights, self.bias)
        return self.initializer
    
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

        weightVelocity, biasVelocity = self.optimizer.update(self.weightsGradient, self.biasGradient)

        self.weights -= learning_rate * weightVelocity
        self.bias -= learning_rate * biasVelocity

        return input_error