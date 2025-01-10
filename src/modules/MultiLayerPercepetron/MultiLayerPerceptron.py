from src.core.layers.FullyActiveConnectedLayer import FullyActiveConnectedLayer as FACLayer
from src.core.functions.ActivationFunctions import ActivationFunctions as activations
from src.core.functions.LossFunctions import LossFunctions as losses
from src.core.regularizations.Regularization import Regularization
from src.core.functions.ActivationFunctionsSelector import ActivationFunctionsSelector as ActivationsSelector
from src.core.functions.LossFunctionsSelector import LossFunctionsSelector as LossSelector
from src.core.regularizations.RegularizationSelector import RegularizationSelector
from .MultiLayerPerceptronSettings import MultiLayerPerceptronSettings as MLPsettings
from .MultiLayerPerceptronCache import MultiLayerPerceptronCache as MLPCache

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
        self.settings = MLPsettings
        self.cache = MLPCache()
        self.initSettings()

    def initSettings(self):
        self.settings["initialization"] = "random"
        self.settings["optimization"] = "momentum"
        self.settings["regularization"] ="none"
        self.settings["loss"] ="mse"
        self.settings["activation"] = "tanh"
        self.settings["verbose"] = "true"

    def setVerbose(self, value):
        if value==True:
            self.verbose = True
            self.settings["verbose"] = "true"
            return
        self.verbose = False
        self.settings["verbose"] = "false"

    
    def setActivationFunction(self, activation):
        activation_prime = activation + "_prime"
        
        if activation not in ActivationsSelector:
            activation = "tanh"
            activation_prime = "tanh_prime"

        self.activation = ActivationsSelector[activation]
        self.activation_prime = ActivationsSelector[activation_prime]

        self.settings["activation"] = activation

    
    def setLossFunction(self, loss):
        loss_prime = loss + "_prime"

        if loss not in LossSelector:
            loss = "mse"
            loss_prime = loss + "_prime"

        self.loss = LossSelector[loss]
        self.loss_prime = LossSelector[loss_prime]

        self.settings["loss"]=loss

    def setRegularization(self, regularization):

        if regularization not in RegularizationSelector:
            regularization = "none"
        
        self.regularization = RegularizationSelector[regularization]

        self.settings["regularization"] = regularization
        
    def setRegularizationParameter(self, newParameter):
        self.regularizationParameter = newParameter
    
    def setInitializer(self, initializer):
        self.initializer = initializer
        self.settings["initialization"] = initializer


    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def add(self, layer):
        self.layers.append(layer)

    def setLayers(self, layersSizes):
        for input_size, output_size in layersSizes:            
            self.add(FACLayer(input_size, output_size,self.initializer, self.optimizer))
        
        self.initializer = self.layers[0].getInitializerName()
        self.optimizer = self.layers[0].getOptimizerName()

    

    def getSettings(self):
        return self.settings

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
        
        if self.verbose:
            print("\n")
            print(self.settings)
            print("\n")

        samples = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]

                for layer in self.layers:
                    output = layer.forward_propagation(output, self.activation)
                
                regError = self.regularization(self.layers, self.regularizationParameter)
                lossError = self.loss(y_train[j], output)
                err += lossError + regError

                error = self.loss_prime(y_train[j], output)
                
                self.cache.errorBySample.append(lossError)
                self.cache.regBySample.append(regError)
                
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, self.activation_prime, self.learningRate)
            err /= samples
            
            self.cache.errorByEpoch.append(err)

            if self.verbose:
                print(f'Epoch {i+1}/{epochs} - Error={err:.6f}')
