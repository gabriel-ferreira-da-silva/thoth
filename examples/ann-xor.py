import numpy as np
from src.modules.ArtificialNeuralNetwork.ArtificialNeuralNetwork import ArtificialNeuralNetwork as Ann 
from src.core.layers.FullyConnectedLayer import FullyConnected as FCLayer
from src.core.layers.ActivationLayer import ActivationLayer
from src.core.functions.ActivationFunctions import ActivationFunctions as activations
from src.core.functions.LossFunctions import LossFunctions as losses
from src.core.metrics.Metrics import Metrics

x_train = np.array( [ [[0,0]] , [[0,1]] ,[[1,0]] , [[1,1]] ])
y_train = np.array( [[[0]] , [[1]], [[1]], [[0]]])


net = Ann()
net.setLayers([(2,3),(3,1)])
net.setLearningRate(0.1)
net.setVerbose(True)
net.fit(x_train, y_train, epochs=1000)
out = net.predict(x_train)
print(out)
