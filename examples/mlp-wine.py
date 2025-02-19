import numpy as np
from src.modules.MultiLayerPerceptron.MultiLayerPerceptron import MultiLayerPerceptron as MLP
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df  = pd.read_csv("/home/gabriel/Desktop/federal/2024.2/deep learning/thoth/datasets/wine/data.csv")
# Assume 'Target' is the column to predict
x = df.drop('quality', axis=1)  # Features
y = df['quality']  # Target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)

print(x_train)

# Convert DataFrame to NumPy arrays before passing them to the MLP
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

print(x_train)
print("shape")
print(x_train.shape)


print("****************")

print(y_train)
print("shape")
print(y_train.shape)



x_train = x_train.reshape(x_train.shape[0],1, x_train.shape[1])  # Add batch and flatten
y_train = y_train.reshape(y_train.shape[0],1)  # Add batch and flatten
x_test = x_test.reshape(x_test.shape[0],1, x_test.shape[1])


print("****************")

print(y_train)
print("shape")
print(y_train.shape)


print("\n\n**************")
print("after reshape")
print(x_train.shape)
print(x_train[0])
print(y_train[0])
print("\n\n******************")

print("\n shape of y")
print(y_train.shape)

# Initialize and train the network

net = MLP()
net.setVerbose(True)
net.setInitializer("he")
net.setOptimizer("momentum")
net.setActivationFunction("relu")
net.setLossFunction("mse")
net.setLearningRate(0.01)
net.setRegularization("none")
net.setLayers([ (11, 10),(10,5),(5,1)])

print(x_train.shape)

EPOCHS = 200
net.fit(x_train, y_train, epochs=EPOCHS)



# Test on 3 samples
#out = np.argmax(net.predict(x_test[:3]), axis=1)
out = net.predict(x_test[:10])
test = y_test[:10]

for i in range(10):
    print("Predicted: ", out[i], " true: ", test[i])

'''

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

'''