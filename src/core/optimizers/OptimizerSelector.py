from .Optimizers import Optimizers

OptimizerSelector = {
    "momentum": lambda: Optimizers.MomentumOptimizer(),  
    "rmsprop": lambda: Optimizers.RMSpropOptimizer(),
    "adagrad": lambda: Optimizers.AdagradOptimizer(),
    "adam": lambda: Optimizers.AdamOptimizer()
}
