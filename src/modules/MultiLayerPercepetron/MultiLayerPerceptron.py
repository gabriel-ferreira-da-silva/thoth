from src.core.layers.FullyActiveConnectedLayer import FullyActiveConnectedLayer as FACLayer
from src.core.functions.ActivationFunctions import ActivationFunctions as activations
from src.core.functions.LossFunctions import LossFunctions as losses
from src.core.regularizations.Regularizarion import Regularization

class MultiLayerPerceptron:
    def __init__(self):
        self.layers = []
        self.loss = losses.mse
        self.loss_prime = losses.mse_prime
        self.verbose = True
        self.activation = activations.relu
        self.activation_prime = activations.relu_prime  
        self.learningRate = 0.01
        self.initializer="random"
        self.regularization = Regularization.none
        self.regularizationParameter = 0.00001

    def setVerbose(self, value):
        if value==True:
            self.verbose = True
            return
        self.verbose = False

    def setRegularizationParameter(self, newParameter):
        self.regularizationParameter = newParameter
    
    def setInitializer(self, initializer):
        self.initializer = initializer

    def setLearningRate(self, value):
        self.learningRate = value

    def add(self, layer):
        self.layers.append(layer)

    def setLayers(self, layersSizes):
        for input_size, output_size in layersSizes:            
            self.add(FACLayer(input_size, output_size,self.initializer))
            
    def setLoss(self, loss, loss_prime):
        self.loss = loss 
        self.loss_prime = loss_prime

    def predict(self, input_data):
        if not self.layers:
            print("No hidden layers are set up")
            return
        
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output, self.activation)
            result.append(output)

        return result
    
    def fit(self, x_train, y_train, epochs):
        if not self.layers:
            print("No hidden layers are set up")
            return
        
        samples = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]

                for layer in self.layers:
                    output = layer.forward_propagation(output, self.activation)
                
                err += self.loss(y_train[j], output) + self.regularization(self.layers, self.regularizationParameter)

                error = self.loss_prime(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, self.activation_prime, self.learningRate)
            err /= samples

            if self.verbose:
                print(f'Epoch {i+1}/{epochs} - Error={err:.6f}')
