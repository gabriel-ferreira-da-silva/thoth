from .LossFunctions import LossFunctions as lf

LossFunctionsSelector ={
    "mean_square_error":lf.mse,
    "mse": lf.mse,
    "mean_square_error_prime": lf.mse_prime,
    "mse_prime": lf.mse_prime,
    
    "log_likelihood": lf.log_likelihood,
    "llh": lf.log_likelihood,
    "log_likelihood_prime": lf.log_likelihood_prime,
    "llh_prime": lf.log_likelihood_prime,
    
    "binary_cross_entropy": lf.binary_cross_entropy,
    "bce": lf.binary_cross_entropy,
    "binary_cross_entropy_prime": lf.binary_cross_entropy_prime,
    "bce_prime": lf.binary_cross_entropy_prime,
    
    "cross_entropy": lf.cross_entropy,
    "ce": lf.cross_entropy,
    "cross_entropy_prime": lf.cross_entropy_prime,
    "ce_prime": lf.cross_entropy_prime,
}