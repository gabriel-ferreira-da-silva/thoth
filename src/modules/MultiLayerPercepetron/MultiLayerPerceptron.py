from src.core.layers.FullyActiveConnectedLayer import FullyActiveConnectedLayer as FACLayer
from src.core.functions.ActivationFunctions import ActivationFunctions as activations
from src.core.functions.LossFunctions import LossFunctions as losses
from src.core.regularizations.Regularizarion import Regularization
from src.core.functions.ActivationFunctionsSelector import ActivationFunctionsSelector as ActivationsSelector
from src.core.functions.LossFunctionsSelector import LossFunctionsSelector as LossSelector
from src.core.regularizations.RegularizationSelector import RegularizationSelector

class MultiLayerPerceptron:
    def __init__(self):
        self.layers = []
        self.loss = losses.mse
        self.loss_prime = losses.mse_prime
        self.verbose = True
        self.activation = activations.tanh
        self.activation_prime = activations.tanh_prime 
        self.learningRate = 0.1
        self.initializer="random"
        self.optimizer="momentum"
        self.regularization = Regularization.none
        self.regularizationParameter = 0.00001

    def setVerbose(self, value):
        if value==True:
            self.verbose = True
            return
        self.verbose = False
    
    def setActivationFunction(self, activation):
        self.activation = ActivationsSelector.get(activation, ActivationsSelector["tanh"])
        prime = activation + "_prime"
        self.activation_prime = ActivationsSelector.get(prime, ActivationsSelector["tanh_prime"])
    
    def setLossFunction(self, loss):
        self.loss = LossSelector.get(loss, LossSelector["mse"])
        prime = loss + "_prime"
        self.loss_prime = LossSelector.get(prime, LossSelector["mse_prime"])

    def setRegularization(self, regularization):
        self.regularization = RegularizationSelector.get(regularization, RegularizationSelector["none"])
        
    def setRegularizationParameter(self, newParameter):
        self.regularizationParameter = newParameter
    
    def setInitializer(self, initializer):
        self.initializer = initializer

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def add(self, layer):
        self.layers.append(layer)

    def setLayers(self, layersSizes):
        for input_size, output_size in layersSizes:            
            self.add(FACLayer(input_size, output_size,self.initializer, self.optimizer))
            
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
