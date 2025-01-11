import numpy as np

class Metrics:

    @staticmethod
    def accuracy(y_true, y_pred):
        
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positive = np.sum(y_pred == 1)
        return true_positive / predicted_positive if predicted_positive != 0 else 0

    @staticmethod
    def recall(y_true, y_pred):
        
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        actual_positive = np.sum(y_true == 1)
        return true_positive / actual_positive if actual_positive != 0 else 0

    @staticmethod
    def f1_score(y_true, y_pred):
        
        precision = Metrics.precision(y_true, y_pred)
        recall = Metrics.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

class RegressionMetrics:
    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(RegressionMetrics.mse(y_true, y_pred))

    @staticmethod
    def r2_score(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)