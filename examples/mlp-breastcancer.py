import numpy as np
from src.modules.MultiLayerPercepetron.MultiLayerPerceptron import MultiLayerPerceptron as MLP
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df  = pd.read_csv("/home/gabriel/Desktop/federal/2024.2/deep learning/thoth/datasets/breast cancer/data.csv")
# Assume 'Target' is the column to predict
x = df.drop(columns=["diagnosis_M", "diagnosis_B"], axis=1)
y = df[["diagnosis_M", "diagnosis_B"]]  # Target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)


# Convert DataFrame to NumPy arrays before passing them to the MLP
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

x_train = x_train.reshape(x_train.shape[0],1, 30)  # Add batch and flatten
x_test = x_test.reshape(x_test.shape[0],1, 30)

# Initialize and train the network
net = MLP()
net.setVerbose(True)
net.setInitializer("random")
net.setOptimizer("rmsprop")
net.setActivationFunction("sigmoid")
net.setLossFunction("mse")
net.setLearningRate(0.1)
net.setRegularization("none")
net.setLayers([ (30, 40),(40,2)])

EPOCHS = 100
net.fit(x_train, y_train, epochs=EPOCHS)

out = net.predict(x_test[:10])

predicted_classes = [np.argmax(o) for o in out]


y_true = np.argmax(y_test[:10], axis=1)

print("******************")
print("y_true")
print(y_true)
print("predicted_classes")
print(predicted_classes)
print("******************")



'''

# Test on 3 samples
#out = np.argmax(net.predict(x_test[:3]), axis=1)
out = net.predict(x_test[:10])

predicted_classes = [np.argmax(o) for o in out]

y_true = np.argmax(y_test[:10], axis=1)

print("\n\npredicted:")
print(predicted_classes)
print("\n\nout:")
print(out)
print("\n\ntrue:")
print(y_true)


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

errors = errors.reshape(3,1, 12000)  # Add batch and flatten
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
