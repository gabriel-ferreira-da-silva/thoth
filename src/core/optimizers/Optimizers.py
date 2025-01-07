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
    
    class AdagradOptimizer(BaseOptimizer):
        def __init__(self, epsilonParameter=0.001):
            self.weightCache = None
            self.biasCache = None
            self.epsilonParameter = epsilonParameter
        
        def initialize(self, weights, bias):
            self.weightCache = np.zeros_like(weights)
            self.biasCache = np.zeros_like(bias)

        def update(self,weights_gradient, bias_gradient):
            self.weightCache += np.square(weights_gradient)
            self.biasCache += np.square(bias_gradient)

            weightsDelta = (weights_gradient / (np.sqrt(self.weightCache) + self.epsilonParameter))
            biasDelta =  (bias_gradient / (np.sqrt(self.biasCache) + self.epsilonParameter))
            return weightsDelta, biasDelta
        
        def getCaches(self):
            return self.weightCache, self.biasCache
    

    class rmspropOptimizer(BaseOptimizer):
        def __init__(self, gamaParameter=0.9):
            self.weightCache = None
            self.biasCache = None
            self.gamaParameter = gamaParameter
        
        def initialize(self, weights, bias):
            self.weightCache = np.zeros_like(weights)
            self.biasCache = np.zeros_like(bias)

        def update(self,weights_gradient, bias_gradient):
            self.weightCache = (self.gamaParameter) * self.weightCache + (1+self.gamaParameter) * np.square(weights_gradient)
            self.biasCache = (self.gamaParameter) *self.biasCache +  (1+self.gamaParameter) *  np.square(bias_gradient)

            weightsDelta = (weights_gradient / (np.sqrt(self.weightCache) + self.gamaParameter))
            biasDelta =  (bias_gradient / (np.sqrt(self.biasCache) + self.gamaParameter))
            return weightsDelta, biasDelta
        
        def getCaches(self):
            return self.weightCache, self.biasCache
    