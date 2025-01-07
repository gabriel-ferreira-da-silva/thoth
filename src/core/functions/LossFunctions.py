import numpy as np

class LossFunctions():

    @staticmethod
    def mse(y_true , y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
    @staticmethod
    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    @staticmethod
    def log_likelihood(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.log(y_pred[np.arange(len(y_true)), y_true.argmax(axis=1)])

    @staticmethod
    def log_likelihood_prime(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        grad = np.zeros_like(y_pred)
        grad[np.arange(len(y_true)), y_true.argmax(axis=1)] = -1 / y_pred[np.arange(len(y_true)), y_true.argmax(axis=1)]
        return grad / len(y_true)

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def binary_cross_entropy_prime(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)

    @staticmethod
    def cross_entropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) 
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    @staticmethod
    def cross_entropy_prime(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / y_pred / y_true.shape[0]
