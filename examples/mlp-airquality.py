import random
import numpy as np
from src.core.metrics.RegressionMetrics import RegressionMetrics
from src.modules.MultiLayerPerceptron.MultiLayerPerceptron import MultiLayerPerceptron as MLP
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from matplotlib.animation import FuncAnimation


df  = pd.read_csv("/home/gabriel/Desktop/federal/2024.2/deep learning/thoth/datasets/airquality/data.csv")
# Assume 'Target' is the column to predict
x = df.drop('NOx(GT)', axis=1)  # Features
y = df['NOx(GT)']  # Target


random_number = random.randint(1, 100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= random_number)


# Convert DataFrame to NumPy arrays before passing them to the MLP
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

print("\n\n**************")
print(x_train.shape)
print(x_train[0])
print(y_train[0])
print("\n\n******************")

x_train = x_train.reshape(x_train.shape[0],1, 12)  # Add batch and flatten
x_test = x_test.reshape(x_test.shape[0],1, 12)

# Initialize and train the network
net = MLP()
net.setVerbose(True)
net.setInitializer("he")
net.setOptimizer("momentum")
net.setActivationFunction("relu")
net.setLossFunction("mse")
net.setLearningRate(0.01)
net.setRegularization("ridge")
net.setLayers([ (12, 20),(20,5),(5,1)])

EPOCHS = 10
net.fit(x_train, y_train, epochs=EPOCHS)


out = net.predict(x_test)

predicted_classes = out

y_true = y_test

mse = RegressionMetrics.mean_squared_error(y_true,  predicted_classes)




'''
print("\n\npredicted:")
print(predicted_classes)
print("\n\nout:")
print(out)
print("\n\ntrue:")
print(y_true)

'''


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

# Example data (replace this with your actual variable)
y = net.cache.regBySample  # Use your data here
y = np.array(y).flatten()
print(y[:5])
size = len(y)
x = np.linspace(0, size, size)  # Generate x values


print("\n\n\n\n**************************")
print(y.shape)
print("\n\n\n\n*******************")

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, size)
ax.set_ylim(min(y), max(y))  # Add some padding around the y values

line, = ax.plot([], [], label="Regulator by sample", color='blue', lw=2)
ax.set_xlabel('Sample')
ax.set_ylabel('Regulator value')
ax.legend()

# Initialize the animation
def init():
    line.set_data([], [])
    return line,

# Update function
def update(frame):
    line.set_data(x[:frame*120], y[:frame*120])  # Plot up to the current frame
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(x) + 1, init_func=init,repeat=True, blit=True, interval=1)
#ani.save('animation.gif', writer='pillow', fps=30)  # Save as GIF

# Show the animation
plt.show()


errors = np.array(net.cache.errorsBySample)

errors = errors.reshape(1, 74850)  # Add batch and flatten



window_size = 100
bab = 1
run = [ ]
xm = [ ]
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

    if bab :
        bab = 0
        run = running_mean
        xm = x_mean

    plt.xlabel('Sample Index')
    plt.ylabel('Error Value')
    plt.legend()
    plt.title(f'Running Mean by sample(Window Size: {window_size})')
    plt.show()


'''

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
print("\nPredicted values: ", predicted_classes[:10])
print("True values: ", y_true[:10])

'''


# Example data (replace this with your actual variable)
y = net.cache.regBySample  # Use your data here
y = np.array(y).flatten()
print(y[:5])
size = len(run)
x = np.linspace(0, size, size)  # Generate x values



# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, size)
ax.set_ylim(min(run), max(run))  # Add some padding around the y values

line, = ax.plot([], [], label="error by sample",color='orange', lw=2)
ax.set_xlabel('Sample')
ax.set_ylabel('Error Value')
ax.legend()

# Initialize the animation
def init1():
    line.set_data([], [])
    return line,

# Update function
def update1(frame):
    line.set_data(xm[:frame*120], run[:frame*120])  # Plot up to the current frame
    return line,

# Create the animation
ani = FuncAnimation(fig, update1, frames=len(xm) + 1, init_func=init1,repeat=True, blit=True, interval=1)
#ani.save('animation.gif', writer='pillow', fps=30)  # Save as GIF

# Show the animation
plt.show()





# Example data: an array of x and y values
x_data = np.linspace(0, 2 * np.pi, 100)
y_data = np.sin(x_data)
'''
# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.5, 1.5)

'''

# Display the animation
plt.show()
