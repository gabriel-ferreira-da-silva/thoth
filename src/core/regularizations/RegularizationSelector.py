from .Regularization import Regularization as reg
RegularizationSelector = {
    "lasso": reg.lasso,
    "ridge": reg.ridge,
    "none": reg.none,
}