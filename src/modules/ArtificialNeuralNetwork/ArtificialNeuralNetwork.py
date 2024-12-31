import numpy as np
from src.core.layers.FullyConnectedLayer import FullyConnected as FCLayer
from src.core.layers.ActivationLayer import ActivationLayer
from src.core.functions.ActivationFunctions import ActivationFunctions as activations
from src.core.functions.LossFunctions import LossFunctions as losses


class ArtificialNeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = losses.mse
        self.loss_prime = losses.mse_prime
        self.verbose = False
        self.activation = activations.tanh
        self.activation_prime = activations.tanh_prime  
        self.learningRate = 0.01

    def setVerbose(self, value):
        if value == True:
            self.verbose = True
            return
        self.verbose = False
    
    def setLearningRate(self, value):
        self.learningRate=value

    def add(self, layer):
        self.layers.append(layer)

    def setLayers(self, layersSizes):
        for layerSize in layersSizes:            
            self.add(FCLayer(layerSize[0],layerSize[1]))
            self.add(ActivationLayer( self.activation, self.activation_prime))

    def setLoss(self, loss,loss_prime):
        self.loss = loss 
        self.loss_prime = loss_prime

    def predict(self, input_data):
        if self.layers == []:
            print("no hidden layers are setted up")
            return
        
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]

            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result
    
    def fit(self , x_train, y_train, epochs):
        
        if self.layers == []:
            print("no hidden layers are setted up")
            return
        
        
        samples = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]

                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                err += self.loss(y_train[j], output)
                error = self.loss_prime(y_train[j] , output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, self.learningRate)
            err /= samples

            if self.verbose:
                print('epoch %d/%d   error=%f' % (i+1, epochs, err))