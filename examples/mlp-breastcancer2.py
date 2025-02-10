import numpy as np

from src.modules.MultiLayerPerceptron.MultiLayerPerceptron import MultiLayerPerceptron as MLP

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from src.core.metrics.ClassificationMetrics import ClassificationMetrics


from matplotlib.animation import FuncAnimation


df  = pd.read_csv("/home/gabriel/Desktop/federal/2024.2/deep learning/thoth/datasets/breast cancer/data.csv")
# Assume 'Target' is the column to predict
x = df.drop(columns=["diagnosis_M", "diagnosis_B"], axis=1)
y = df[["diagnosis_M", "diagnosis_B"]]  # Target

import random

random_number = random.randint(1, 100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= random_number)


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
net.setInitializer("normal")
net.setOptimizer("adam")
net.setActivationFunction("sigmoid")
net.setLossFunction("mse")
net.setLearningRate(0.01)
net.setRegularization("ridge")
net.setLayers([ (30, 40),(40,2)])

EPOCHS = 10
net.fit(x_train, y_train, epochs=EPOCHS)

out = net.predict(x_test)

predicted_classes = [np.argmax(o) for o in out]


y_true = np.argmax(y_test, axis=1)

print("******************")
print("y_true")
print(y_true)
print("predicted_classes")
print(predicted_classes)
print("******************")

acc = ClassificationMetrics.accuracy(y_true,  predicted_classes)
f1 = ClassificationMetrics.f1_score(y_true,  predicted_classes)
recall = ClassificationMetrics.recall(y_true, predicted_classes)
precision = ClassificationMetrics.precision(y_true, predicted_classes)


print("************************************")
print("acc: ", acc)
print("f1: ", f1)
print("recall: ", recall)
print("precision: ", precision)
print("************************************")




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
plt.plot(x, y, label="error by epoch")
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend()
plt.show()

y = net.cache.errorLossBySample
size = len(y)
x = np.linspace(0, size, size)  # 100 points between 0 and 10

plt.figure(figsize=(8, 6))
plt.plot(x, y, label="error by sample")

plt.xlabel('sample')
plt.ylabel('error')
plt.legend()
plt.show()

y = net.cache.regBySample
size = len(y)
x = np.linspace(0, size, size)  # 100 points between 0 and 10

plt.figure(figsize=(8, 6))
plt.plot(x, y, label="regulator value")

plt.xlabel('sample')
plt.ylabel('regulator value')
plt.legend()

plt.show()


errors = np.array(net.cache.errorsBySample)
print(errors.shape)

errors = errors.reshape(1, 9100)  # Add batch and flatten
print(errors.shape)

print(len(errors))
print(len(errors))
print(len(errors))
print(len(errors))

window_size = 100
running_mean = None
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
    #plt.plot(x, y, label="Error by sample", alpha=0.5)
    plt.plot(x_mean, running_mean, label=f"Running Mean (Window={window_size})", color='red')

    plt.xlabel('Sample Index')
    plt.ylabel('Error Value')
    plt.legend()
    plt.title(f'Running Mean by sample(Window Size: {window_size})')
    plt.show()


def plot_weights(weights_by_layer, opt):
    means = [np.mean(weights) for weights in weights_by_layer]  # Mean for each layer
    std_devs = [np.std(weights) for weights in weights_by_layer]  # Standard deviation for each layer

    # X-axis labels for each layer
    layers = [f'Layer {i+1}' for i in range(len(weights_by_layer))]

    # Create the bar plot with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(layers, means, yerr=std_devs, capsize=5, color='skyblue', edgecolor='black')
    
    # Adding labels and title
    plt.xlabel('Layers')
    plt.ylabel('Mean ' +opt+' Value')
    plt.title('Mean and Standard Deviation of '+opt + ' by Layer')
    plt.tight_layout()
    plt.show()

# Example usage:
weights_by_layer = net.getWeights()  # Assuming your model object has the getWeights method
plot_weights(weights_by_layer, "weight")
plot_weights(net.getBias(), "bias")
print("\nPredicted values: ", predicted_classes)
print("True values: ", y_true)





errors = np.array(net.cache.errorsBySample)
errors = errors.reshape(1, 9100)  # Add batch and flatten

window_size = 100
running_mean = None

# Prepare the figure and axis for plotting
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], label=f"Running Mean (Window={window_size})", color='red')
ax.set_xlim(0, len(errors[0]))  # x-axis for sample index
ax.set_ylim(np.min(errors), np.max(errors))  # y-axis for error values
ax.set_xlabel('Sample Index')
ax.set_ylabel('Error Value')
ax.legend()
ax.set_title(f'Running Mean by Sample (Window Size: {window_size})')

# Initialize function for animation
def init():
    line.set_data([], [])
    return line,

# Update function for animation
def update(frame):
    global running_mean
    y = np.array(errors[0]).flatten()  # Ensure y is a flat 1D array
    size = len(y)
    x = np.arange(size)

    # Calculate running mean considering the last 'window_size' values using a moving average
    if frame >= window_size:
        running_mean = np.convolve(y[:frame], np.ones(window_size) / window_size, mode='valid')
        x_mean = np.arange(window_size - 1, frame)  # Adjust x-axis for the reduced data
    else:
        running_mean = y[:frame]  # If fewer than window_size points, just plot the original data
        x_mean = x[:frame]  # Adjust x-axis for the actual frame range

    # Update the line with the running mean
    line.set_data(x_mean, running_mean)
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(errors[0]), init_func=init, blit=True, interval=1)

# Show the animation
plt.show()