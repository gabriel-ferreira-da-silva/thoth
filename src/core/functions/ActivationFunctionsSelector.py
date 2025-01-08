from .ActivationFunctions import ActivationFunctions as af

ActivationFunctionsSelector = {
    "linear" : af.linear,
    "linear_prime": af.linear_prime,
    "relu": af.relu,
    "relu_prime" : af.relu_prime,
    "softplus": af.softplus,
    "softplus_prime": af.softplus_prime,
    "sigmoid" : af.sigmoid,
    "sigmoid_prime": af.sigmoid_prime,
    "softmax": af.softmax,
    "softmax_prime": af.softmax_prime,
    "tanh": af.tanh,
    "tanh_prime": af.tanh_prime,
}