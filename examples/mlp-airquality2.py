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

EPOCHS = 1000
net.fit(x_train, y_train, epochs=EPOCHS)