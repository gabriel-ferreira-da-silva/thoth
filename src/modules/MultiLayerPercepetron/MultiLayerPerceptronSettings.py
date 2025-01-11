from .RestrictedDict import RestrictedDict

MultiLayerPerceptronSettings = RestrictedDict({
    "initialization": None,
    "optimization": None,
    "regularization": None,
    "loss": None,
    "activation": None, 
    "verbose": None,
})