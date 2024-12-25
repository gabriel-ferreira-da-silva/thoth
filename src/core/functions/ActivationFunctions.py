
import numpy as np

class ActivationFunctions():
    
    def tanh(x):
        return np.tanh(x)
    
    def tanh_prime(x):
        return 1 - np.tanh(x) ** 2