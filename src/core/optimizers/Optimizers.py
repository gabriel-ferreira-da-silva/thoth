import numpy as np
from .BaseOptimizer import BaseOptimizer

class Optimizers:

    class MomentumOptimizer(BaseOptimizer):
        def __init__(self, learning_rate=0.01, beta=0.9):
            self.learning_rate = learning_rate
            self.beta = beta
            self.velocities_weights = []
            self.velocities_bias = []

        def initialize(self, layers):
            self.velocities_weights = [np.zeros_like(layer.getWeights()) for layer in layers]
            self.velocities_bias = [np.zeros_like(layer.getBias()) for layer in layers]

        def update(self, layers):
            for i, layer in enumerate(layers):
                gradients_w = layer.getWeightGradients()
                gradients_b = layer.getBiasGradients()
                
                if gradients_w is None or gradients_b is None:
                    raise ValueError(f"Gradients for layer {i} were not set during backpropagation!")

                self.velocities_weights[i] = self.beta * self.velocities_weights[i] + (1 - self.beta) * gradients_w
                self.velocities_bias[i] = self.beta * self.velocities_bias[i] + (1 - self.beta) * gradients_b

                newWeights = layer.getWeights() - self.learning_rate * self.velocities_weights[i]
                newBias = layer.getBias() - self.learning_rate * self.velocities_bias[i]

                layer.setWeights(newWeights)
                layer.setBias(newBias)

