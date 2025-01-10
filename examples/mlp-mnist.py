import numpy as np
from src.modules.MultiLayerPercepetron.MultiLayerPerceptron import MultiLayerPerceptron as MLP
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess input data
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)  # Add batch and flatten
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)

# One-hot encode output labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Initialize the network
net = MLP()
net.setVerbose(True)
net.setInitializer("random")
net.setOptimizer("momentum")
net.setActivationFunction("sigmoid")
net.setLossFunction("mse")
net.setRegularization("ridge")
net.setLayers([(28*28, 100), (100, 50), (50, 10)])

# Train on a subset of data
net.fit(x_train[:1000], y_train[:1000], epochs=50)

# Test on 3 samples
#out = np.argmax(net.predict(x_test[:3]), axis=1)
out = net.predict(x_test[:3])

predicted_classes = [np.argmax(o) for o in out]


y_true = np.argmax(y_test[:3], axis=1)

print("\nPredicted values: ", predicted_classes)
print("True values: ", y_true)