import numpy as np

from src.modules.SimpleNeuralNetwork.SimpleNeuralNetwork import SimpleNeuralNetwork 
from src.core.layers.FullyConnectedLayer import FullyConnected as FCLayer
from src.core.layers.ActivationLayer import ActivationLayer
from src.core.functions.ActivationFunctions import ActivationFunctions as activations
from src.core.functions.LossFunctions import LossFunctions as losses
from src.core.metrics.ClassificationMetrics import ClassificationMetrics as Metrics

# Dados XOR (problema de classificação)
x_train = np.array( [ [[0,0]] , [[0,1]] ,[[1,0]] , [[1,1]] ])
y_train = np.array( [[[0]] , [[1]], [[1]], [[0]]])

# Inicializar rede neural
net = SimpleNeuralNetwork()

net.add(FCLayer(2, 3))
net.add(ActivationLayer(activations.tanh, activations.tanh_prime))

net.add(FCLayer(3, 1))
net.add(ActivationLayer(activations.tanh, activations.tanh_prime))

# Configurar função de perda
net.use(losses.mse, losses.mse_prime)

# Treinamento
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# Previsão
out = net.predict(x_train)
predicted_classes = np.round(out).astype(int).flatten()  # Aproxima os valores para 0 ou 1
y_true = y_train.flatten()

print("\nPredicted values: ", predicted_classes)
print("True values: ", y_true)

# Métricas de classificação
accuracy = Metrics.accuracy(y_true, predicted_classes)
precision = Metrics.precision(y_true, predicted_classes)
recall = Metrics.recall(y_true, predicted_classes)
f1_score = Metrics.f1_score(y_true, predicted_classes)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
