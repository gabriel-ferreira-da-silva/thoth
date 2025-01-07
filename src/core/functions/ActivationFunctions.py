
import numpy as np

class ActivationFunctions():

#Função de ativação para regressão:

    @staticmethod
    def linear(x):
        return x
    @staticmethod
    def linear_prime(x):
        return np.ones_like(x)
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    @staticmethod
    def relu_prime(x):
        return np.where(x > 0, 1, 0)
    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))
    @staticmethod
    def softplus_prime(x):
        return 1 / (1 + np.exp(-x))
    

#Função de ativação para classificação:    

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def sigmoid_prime(x):
        return (1 / (1 + np.exp(-x))) * (1 - 1 / (1 + np.exp(-x)))
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    @staticmethod
    def softmax_prime(x):
        s = ActivationFunctions.softmax(x).reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)   
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    @staticmethod
    def tanh_prime(x):
        return 1 - np.tanh(x) ** 2