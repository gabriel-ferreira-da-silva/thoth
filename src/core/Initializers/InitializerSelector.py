from .Initializer import Initializer

InitializerSelector = {
    "cosine": Initializer.cosine,
    "random":Initializer.random,
    "uniform":Initializer.random,
}