# Implementing the Gaussian Process model
# Import a covariance function (kernel), use it to calculate the various covariance matrices,
# and then use the covariance matrices to calculate the expected value and covariance of the predictive
# distribution. From there, we can sample points to make predictions.
import torch
import gpytorch
import numpy as np

from utils.deep_kernel_utils import(
    Deep_Transform
)

# Define own GP model
# Not sure if this is as general as I would like, as depending on kernel
# this might need to change.
class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, marginal_likelihood, mean, covar):
        super(GP, self).__init__(train_x, train_y, marginal_likelihood)
        self.transform = Deep_Transform(input_dim=train_x.size(dim=-1))
        self.mean_module = mean
        self.covar_module = covar   # This is the prior kernel
        
    def forward(self, x):
        # Pass data through transformation
        x_transform = self.transform(x)
        x_transform = self.scale_to_bounds(x_transform)  # Make the NN values "nice"

        x_mean = self.mean_module(x_transform)      # Compute mean
        x_covar = self.covar_module(x_transform)    # and covariance
        return gpytorch.distributions.MultivariateNormal(x_mean, x_covar)   # Return multivariate gaussian defined by mean and covariance