import numpy as np
from src.modules.MultiLayerPerceptron.MultiLayerPerceptron import MultiLayerPerceptron as MLP
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


EPOCHS = 15
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
net.setRegularization("none")
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
errors = errors.reshape(10,15000,1)
print(errors.shape)


window_size = 100

for error in errors:
    y = np.array(error).flatten()  # Ensure y is a flat 1D array
    size = len(y)
    x = np.arange(size)  

    # Calculate running mean considering the last 10 values using a moving average
    if size >= window_size:
        running_mean = np.convolve(y, np.ones(window_size) / window_size, mode='valid')
        x_mean = np.arange(window_size - 1, size)  # Adjust x-axis for the reduced data
    else:
        running_mean = y  # If there are fewer than 10 points, just plot the original data
        x_mean = x

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Error by sample", alpha=0.5)
    plt.plot(x_mean, running_mean, label=f"Running Mean (Window={window_size})", color='red')

    plt.xlabel('Sample Index')
    plt.ylabel('Error Value')
    plt.legend()
    plt.title(f'Error and Running Mean (Window Size: {window_size})')
    plt.show()


def plot_weights(weights_by_layer):
    means = [np.mean(weights) for weights in weights_by_layer]  # Mean for each layer
    std_devs = [np.std(weights) for weights in weights_by_layer]  # Standard deviation for each layer

    # X-axis labels for each layer
    layers = [f'Layer {i+1}' for i in range(len(weights_by_layer))]

    # Create the bar plot with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(layers, means, yerr=std_devs, capsize=5, color='skyblue', edgecolor='black')
    
    # Adding labels and title
    plt.xlabel('Layers')
    plt.ylabel('Mean Weight Value')
    plt.title('Mean and Standard Deviation of Weights by Layer')
    plt.tight_layout()
    plt.show()

# Example usage:
weights_by_layer = net.getWeights()  # Assuming your model object has the getWeights method
plot_weights(weights_by_layer)
plot_weights(net.getBias())
print("\nPredicted values: ", predicted_classes)
print("True values: ", y_true)
