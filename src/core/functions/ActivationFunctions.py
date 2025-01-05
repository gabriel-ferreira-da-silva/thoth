
import numpy as np

class ActivationFunctions():
    
    def tanh(x):
        return np.tanh(x)
    
    def tanh_prime(x):
        return 1 - np.tanh(x) ** 2
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_prime(x):
        return (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x)))