import numpy as np

from src.modules.SimpleNeuralNetwork.SimpleNeuralNetwork import SimpleNeuralNetwork 
from src.core.layers.FullyConnectedLayer import FullyConnected as FCLayer
from src.core.layers.ActivationLayer import ActivationLayer
from src.core.functions.ActivationFunctions import ActivationFunctions as activations
from src.core.functions.LossFunctions import LossFunctions as losses

x_train = np.array( [ [[0,0]] , [[0,1]] ,[[1,0]] , [[1,1]] ])
y_train = np.array( [[[0]] , [[1]], [[1]], [[0]]])

net = SimpleNeuralNetwork()

net.add(FCLayer(2,3))
net.add(ActivationLayer( activations.tanh, activations.tanh_prime))

net.add(FCLayer(3,1))
net.add(ActivationLayer( activations.tanh, activations.tanh_prime))

net.use(losses.mse , losses.mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

out = net.predict(x_train)

print(out)