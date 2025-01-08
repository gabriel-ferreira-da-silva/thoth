import numpy as np

class Regularization:
    @staticmethod
    def lasso(layers, lambda_reg):
        l1_penalty = 0
        for layer in layers:
            if hasattr(layer, "getWeights"):
                weights = layer.getWeights()
                l1_penalty += lambda_reg * np.sum(np.abs(weights))
        return l1_penalty

    @staticmethod
    def ridge(layers, lambda_reg):
        l2_penalty = 0
        for layer in layers:
            weights = layer.getWeights()  
            l2_penalty += lambda_reg * np.sum(weights ** 2)
        return l2_penalty


    @staticmethod
    def none(layer, lambda_reg):
        return 0
