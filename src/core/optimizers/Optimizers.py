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
    

    class RMSpropOptimizer(BaseOptimizer):
        def __init__(self, gamaParameter=0.9):
            self.weightCache = None
            self.biasCache = None
            self.gamaParameter = gamaParameter
        
        def initialize(self, weights, bias):
            self.weightCache = np.zeros_like(weights)
            self.biasCache = np.zeros_like(bias)

        def update(self,weights_gradient, bias_gradient):
            self.weightCache = (self.gamaParameter) * self.weightCache + (1-self.gamaParameter) * np.square(weights_gradient)
            self.biasCache = (self.gamaParameter) *self.biasCache +  (1-self.gamaParameter) *  np.square(bias_gradient)

            weightsDelta = (weights_gradient / (np.sqrt(self.weightCache) + self.gamaParameter))
            biasDelta =  (bias_gradient / (np.sqrt(self.biasCache) + self.gamaParameter))
            return weightsDelta, biasDelta
        
        def getCaches(self):
                return self.weightCache, self.biasCache
        
    class AdamOptimizer(BaseOptimizer):
        def __init__(self, learningRate=0.001, beta1Parameter=0.9, beta2Parameter=0.999, epsilonParameter=1e-8):
            self.learningRate = learningRate
            self.beta1Parameter = beta1Parameter
            self.beta2Parameter = beta2Parameter
            self.epsilonParameter = epsilonParameter
            self.weightCache = None
            self.biasCache = None
            self.weightVelocity = None
            self.biasVelocity = None
            self.timeStep = 0

        def initialize(self, weights, bias):
            self.timeStep = 0
            self.weightCache = np.zeros_like(weights)
            self.biasCache = np.zeros_like(bias)
            self.weightVelocity = np.zeros_like(weights)
            self.biasVelocity = np.zeros_like(bias)

        def update(self, weights_gradient, bias_gradient):
            """Calculate weight and bias updates without directly modifying weights."""
            self.timeStep += 1

            # Update biased first moment estimate
            self.weightVelocity = self.beta1Parameter * self.weightVelocity + (1 - self.beta1Parameter) * weights_gradient
            self.biasVelocity = self.beta1Parameter * self.biasVelocity + (1 - self.beta1Parameter) * bias_gradient

            # Update biased second raw moment estimate
            self.weightCache = self.beta2Parameter * self.weightCache + (1 - self.beta2Parameter) * np.square(weights_gradient)
            self.biasCache = self.beta2Parameter * self.biasCache + (1 - self.beta2Parameter) * np.square(bias_gradient)

            # Bias-corrected first and second moment estimates
            corrected_weightVelocity = self.weightVelocity / (1 - self.beta1Parameter**self.timeStep)
            corrected_biasVelocity = self.biasVelocity / (1 - self.beta1Parameter**self.timeStep)
            corrected_weightCache = self.weightCache / (1 - self.beta2Parameter**self.timeStep)
            corrected_biasCache = self.biasCache / (1 - self.beta2Parameter**self.timeStep)


            weightsDelta =  corrected_weightVelocity / (np.sqrt(corrected_weightCache) + self.epsilonParameter)
            biasDelta = corrected_biasVelocity / (np.sqrt(corrected_biasCache) + self.epsilonParameter)

            return weightsDelta, biasDelta

        def getCaches(self):
            return self.weightCache, self.biasCache