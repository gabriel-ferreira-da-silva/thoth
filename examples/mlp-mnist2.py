import numpy as np
from src.modules.MultiLayerPercepetron.MultiLayerPerceptron import MultiLayerPerceptron as MLP
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


EPOCHS = 5
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
net.setActivationFunction("tanh")
net.setLossFunction("mse")
net.setRegularization("lasso")
net.setLayers([ (28*28, 100),  (100, 50),  (50, 10)])

# Train on a subset of data
net.fit(x_train[:1000], y_train[:1000], epochs=EPOCHS)

# Test on 3 samples
#out = np.argmax(net.predict(x_test[:3]), axis=1)
out = net.predict(x_test[:3])

predicted_classes = [np.argmax(o) for o in out]


y_true = np.argmax(y_test[:3], axis=1)

x = np.linspace(0, EPOCHS, EPOCHS)  # 100 points between 0 and 10

y = net.cache.errorByEpoch
plt.figure(figsize=(8, 6))
plt.plot(x, y)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend("error by epoch")

plt.show()

y = net.cache.errorLossBySample
size = len(y)
x = np.linspace(0, size, size)  # 100 points between 0 and 10

plt.figure(figsize=(8, 6))
plt.plot(x, y)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend("error by sample")
plt.show()

y = net.cache.regBySample
size = len(y)
x = np.linspace(0, size, size)  # 100 points between 0 and 10

plt.figure(figsize=(8, 6))
plt.plot(x, y)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend("error by sample")

plt.show()


errors = np.array(net.cache.errorsBySample)
errors = errors.reshape(10,5000,1)
print(errors.shape)
for error in errors:
    y = error
    print(len(y))
    size = len(y)
    x = np.linspace(0, size, size)  # 100 points between 0 and 10

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend("error by sample")

    plt.show()


print("\nPredicted values: ", predicted_classes)
print("True values: ", y_true)
