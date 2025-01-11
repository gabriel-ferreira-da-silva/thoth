import numpy as np
from src.modules.ArtificialNeuralNetwork.ArtificialNeuralNetwork import ArtificialNeuralNetwork as Ann 
from src.core.metrics.RegressionMetrics import RegressionMetrics as Metrics

x_train = np.array( [ [[0,0]] , [[0,1]] ,[[1,0]] , [[1,1]] ])
y_train = np.array( [[[0]] , [[1]], [[1]], [[0]]])

net = Ann()
net.setLayers([(2,3),(3,1)])
net.setLearningRate(0.1)
net.setVerbose(True)
net.fit(x_train, y_train, epochs=1000)
out = net.predict(x_train)
print(out)

# Metrics
mae = Metrics.mean_absolute_error(y_train, out)
mse = Metrics.mean_squared_error(y_train, out)
rmse = Metrics.root_mean_squared_error(y_train, out)
r2 = Metrics.r2_score(y_train, out)

print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")
