import numpy as np

class BaseOptimizer:
    def __init__(self, learning_rate, beta):
        self.learning_rate = None
        self.beta = None
        self.velocities = None

    def initialize(self, layers):
        raise NotImplementedError

    def update(self, layers):
        raise NotImplementedError
