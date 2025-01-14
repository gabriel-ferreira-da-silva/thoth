from .Initializer import Initializer

InitializerSelector = {
    "cosine": Initializer.cosine,
    "random":Initializer.random,
    "uniform":Initializer.uniform,
    "he":Initializer.he,
    "normal": Initializer.normal,
    "xavier": Initializer.normal,
}

