
import numpy as np

class ActivationFunctions():

#Função de ativação para regressão:

    def linear(x):
        return x
    
    def linear_prime(x):
        return np.ones_like(x)
    
    def relu(x):
        return np.maximum(0, x)
    
    def relu_prime(x):
        return np.where(x > 0, 1, 0)
    
    def softplus(x):
        return np.log(1 + np.exp(x))
    
    def softplus_prime(x):
        return 1 / (1 + np.exp(-x))
    

#Função de ativação para classificação:    

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_prime(x):
        return (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x)))
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def softmax_prime(x):
        s = ActivationFunctions.softmax(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)   
    
    def tanh(x):
        return np.tanh(x)
    
    def tanh_prime(x):
        return 1 - np.tanh(x) ** 2