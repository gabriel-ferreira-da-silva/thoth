import numpy as np

class RegressionMetrics:

#MAE
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

#MSE
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

#RMSE
    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        mse = RegressionMetrics.mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)

#R²
    @staticmethod
    def r2_score(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true))**2)
        ss_residual = np.sum((y_true - y_pred)**2)
        return 1 - (ss_residual / ss_total)
