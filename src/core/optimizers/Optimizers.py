import numpy as np
from .BaseOptimizer import BaseOptimizer

class Optimizers:
    class MomentumOptimizer(BaseOptimizer):
        def __init__(self, parameters=None):
            self.optimizationParameter = parameters[0] if parameters else 0.9
            self.weightVelocity = None
            self.biasVelocity = None
        
        def initialize(self, weights, bias):
            self.weightVelocity = np.zeros_like(weights)
            self.biasVelocity = np.zeros_like(bias)

        def update(self, weights_gradient, bias_gradient):
            self.weightVelocity = (
                self.optimizationParameter * self.weightVelocity
                + (1 - self.optimizationParameter) * weights_gradient
            )
            self.biasVelocity = (
                self.optimizationParameter * self.biasVelocity
                + (1 - self.optimizationParameter) * bias_gradient
            )
            return self.weightVelocity, self.biasVelocity

    class AdagradOptimizer(BaseOptimizer):
        def __init__(self, parameters=None):
            self.epsilonParameter = parameters[0] if parameters else 1e-8
            self.weightCache = None
            self.biasCache = None
        
        def initialize(self, weights, bias):
            self.weightCache = np.zeros_like(weights)
            self.biasCache = np.zeros_like(bias)

        def update(self, weights_gradient, bias_gradient):
            self.weightCache += np.square(weights_gradient)
            self.biasCache += np.square(bias_gradient)

            weightsDelta = weights_gradient / (np.sqrt(self.weightCache) + self.epsilonParameter)
            biasDelta = bias_gradient / (np.sqrt(self.biasCache) + self.epsilonParameter)
            return weightsDelta, biasDelta

    class RMSpropOptimizer(BaseOptimizer):
        def __init__(self, parameters=None):
            self.gammaParameter = parameters[0] if parameters else 0.9
            self.epsilonParameter = parameters[1] if parameters else 1e-8
            self.weightCache = None
            self.biasCache = None

        def initialize(self, weights, bias):
            self.weightCache = np.zeros_like(weights)
            self.biasCache = np.zeros_like(bias)

        def update(self, weights_gradient, bias_gradient):
            self.weightCache = (
                self.gammaParameter * self.weightCache
                + (1 - self.gammaParameter) * np.square(weights_gradient)
            )
            self.biasCache = (
                self.gammaParameter * self.biasCache
                + (1 - self.gammaParameter) * np.square(bias_gradient)
            )

            weightsDelta = weights_gradient / (np.sqrt(self.weightCache) + self.epsilonParameter)
            biasDelta = bias_gradient / (np.sqrt(self.biasCache) + self.epsilonParameter)
            return weightsDelta, biasDelta

    class AdamOptimizer(BaseOptimizer):
        def __init__(self, parameters=None):
            self.beta1Parameter = parameters[0] if parameters else 0.9
            self.beta2Parameter = parameters[1] if parameters else 0.999
            self.epsilonParameter = parameters[2] if parameters else 1e-8
            self.timeStep = 0
            self.weightCache = None
            self.biasCache = None
            self.weightVelocity = None
            self.biasVelocity = None

        def initialize(self, weights, bias):
            self.timeStep = 0
            self.weightCache = np.zeros_like(weights)
            self.biasCache = np.zeros_like(bias)
            self.weightVelocity = np.zeros_like(weights)
            self.biasVelocity = np.zeros_like(bias)

        def update(self, weights_gradient, bias_gradient):
            self.timeStep += 1
            self.weightVelocity = (
                self.beta1Parameter * self.weightVelocity
                + (1 - self.beta1Parameter) * weights_gradient
            )
            self.biasVelocity = (
                self.beta1Parameter * self.biasVelocity
                + (1 - self.beta1Parameter) * bias_gradient
            )

            self.weightCache = (
                self.beta2Parameter * self.weightCache
                + (1 - self.beta2Parameter) * np.square(weights_gradient)
            )
            self.biasCache = (
                self.beta2Parameter * self.biasCache
                + (1 - self.beta2Parameter) * np.square(bias_gradient)
            )

            corrected_weightVelocity = self.weightVelocity / (
                1 - self.beta1Parameter**self.timeStep
            )
            corrected_biasVelocity = self.biasVelocity / (
                1 - self.beta1Parameter**self.timeStep
            )
            corrected_weightCache = self.weightCache / (
                1 - self.beta2Parameter**self.timeStep
            )
            corrected_biasCache = self.biasCache / (
                1 - self.beta2Parameter**self.timeStep
            )

            weightsDelta = corrected_weightVelocity / (
                np.sqrt(corrected_weightCache) + self.epsilonParameter
            )
            biasDelta = corrected_biasVelocity / (
                np.sqrt(corrected_biasCache) + self.epsilonParameter
            )

            return weightsDelta, biasDelta
