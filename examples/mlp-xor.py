import numpy as np
from src.core.metrics.Metrics import Metrics

from src.modules.MultiLayerPercepetron.MultiLayerPerceptron import MultiLayerPerceptron as MLP
x_train = np.array( [ [[0,0]] , [[0,1]] ,[[1,0]] , [[1,1]] ])
y_train = np.array( [[[0]] , [[1]], [[1]], [[0]]])

net = MLP()
net.setLayers([(2,3),(3,1)])
net.setLearningRate(0.1)
net.setVerbose(True)
net.fit(x_train, y_train, epochs=1000)
out = net.predict(x_train)
print(out)
