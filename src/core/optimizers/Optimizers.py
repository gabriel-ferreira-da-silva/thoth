import numpy as np
from .BaseOptimizer import BaseOptimizer

class Optimizers():
    class MomentumOptimizer(BaseOptimizer):
        def __init__(self, optimizationParameter=0.9):
            self.weightVelocity = None
            self.biasVelocity = None
            self.optimizationParameter = optimizationParameter
        
        def initialize(self, weights, bias):
            self.weightVelocity = np.zeros_like(weights)
            self.biasVelocity = np.zeros_like(bias)

        def update(self,weights_gradient, bias_gradient):
            self.weightVelocity = self.optimizationParameter * self.weightVelocity + (1 - self.optimizationParameter) * weights_gradient
            self.biasVelocity = self.optimizationParameter * self.biasVelocity + (1 - self.optimizationParameter) * bias_gradient
            return self.weightVelocity, self.biasVelocity
        
        def getVelocities(self):
            return self.weightVelocity, self.biasVelocity