import torch
import gpytorch
import numpy as np
from matplotlib import pyplot as plt

import os
import sys

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(module_path)

from tqdm import tqdm   # For progress bars

from utils.loss_utils import(
    Generic_Loss,
    mse_loss,
)
from utils.data_utils import (
    gen_interior_points_rectangle, data_square, data_loader_square
)
from utils.finite_diff import (
    finite_diff
)
from utils.laplacian_utils import (
    train_no_batches_rectangle, train, rectangle_loss
)
from models.NeuralPDE import (
    VanillaMLP
)
from experiments.laplacian_dirichlet_rectangle import (
    rectangle_train
)
from utils.plot_utils import (
    plot_solution, make_gif
)
from Regression.regression_data import(
    gen_regression_data,
)

trained_PINN, x_train, y_train = rectangle_train()
trained_PINN.eval()
# Generate new test points
h = 0.0201
length_x = 2.0
length_y = 2.0
x_vals = np.arange(0, length_x + 0.001, h)
y_vals = np.arange(0, length_y + 0.001, h)

u_square = finite_diff(x_vals, y_vals)

x_vals, y_vals = torch.tensor(x_vals).float(), torch.tensor(y_vals).float()
x_vals, y_vals = torch.meshgrid(x_vals, y_vals, indexing="ij")
x_vals_flat = x_vals.flatten()
y_vals_flat = y_vals.flatten()
xy_vals = torch.cat((x_vals_flat.unsqueeze(1), y_vals_flat.unsqueeze(1)), dim=-1)
# Generate solutions using the trained_PINN
u_gen = []
with torch.no_grad():
    for z in xy_vals:
        u = trained_PINN(z)
        u_gen.append(u)
        
u_true = torch.tensor(u_square).float().flatten()
u_gen = torch.tensor(u_gen).float()

# Define a GP with of points u_gen(x, y) - u_true(x, y). Should have 0 mean and should fit a covariance to it.
class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean, covar):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = covar  # This is the prior kernel
        self.distribution = gpytorch.distributions.MultivariateNormal   # Set distribution to multivariate normal
        
    def forward(self, x):
        # Pass data through transformation
        x_mean = self.mean_module(x)      # Compute mean
        x_covar = self.covar_module(x)    # and covariance
        return self.distribution(x_mean, x_covar)   # Return multivariate gaussian defined by mean and covariance

u_zeroed = u_true - u_gen   # Subtract mean (PINN)
# print(u_zeroed)
# input("Press Enter to continue...")
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) # RBF Kernel
# mean = gpytorch.means.ConstantMean()    # Some mean
mean = gpytorch.means.ZeroMean()
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GP(xy_vals, u_zeroed, likelihood, mean=mean, covar=kernel)

# Training
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
marginal_ll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(40):
    optimizer.zero_grad()
    output = model(xy_vals)
    loss = -marginal_ll(output, u_zeroed)
    loss.backward()
    optimizer.step()
    if (i + 1) % 10 == 0:
        print(f'Iter {i+1}/{100} - Loss: {loss.item()}, -Lengthscale: {model.covar_module.base_kernel.lengthscale.item()}, noise: {model.likelihood.noise.item()}')
# Predictions
model.eval()
likelihood.eval()

# Generate new test points
h = 0.0301
length_x = 2.0
length_y = 2.0
x_star = np.arange(0, length_x + 0.001, h)
y_star = np.arange(0, length_y + 0.001, h)

u_star = finite_diff(x_star, y_star)
u_star = torch.tensor(u_star).float().flatten()

x_star, y_star = torch.tensor(x_star).float(), torch.tensor(y_star).float()
x_star, y_star = torch.meshgrid(x_star, y_star, indexing="ij")
x_star_flat = x_star.flatten()
y_star_flat = y_star.flatten()
xy_star = torch.cat((x_star_flat.unsqueeze(1), y_star_flat.unsqueeze(1)), dim=-1)
# Generate solutions using the trained_PINN
# Can also try reusing training points, since otherwise GP won't fit correct confidence bound on the PINN 
# as PINN will generate points for the test that might not align with the 'mean' of the fitted GP

u_mean = []
with torch.no_grad():
    for z in xy_star:
        u = trained_PINN(z)
        u = u.detach()
        u_mean.append(u)
u_mean = torch.tensor(u_mean).float()

print("NUM TEST POINTS:", u_mean.shape)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(xy_star))
    pred_mean = u_mean  # Try using just the neural network as the predictive mean as well
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    lower_surface, upper_surface = lower + u_mean, upper + u_mean
    lower_surface = lower_surface.reshape(x_star.shape).numpy()
    upper_surface = upper_surface.reshape(x_star.shape).numpy()
    # Plot training data as black stars
    # ax.scatter(x_vals, y_vals, u_true, c="k", marker="*", label="Observed Data")
    # Plot predictive means as blue line
    ax.plot_surface(x_star.numpy(), y_star.numpy(), pred_mean.reshape(x_star.shape).numpy(), alpha=0.5, color="green", label="Predictive Mean")
    ax.plot_surface(x_star.numpy(), y_star.numpy(), u_star.reshape(x_star.shape).numpy(), alpha=0.5, color="purple", label="Finite Diff")
    ax.plot_surface(x_star.numpy(), y_star.numpy(), lower_surface, alpha=0.5, color="blue", label="Lower Bound")
    ax.plot_surface(x_star.numpy(), y_star.numpy(), upper_surface, alpha=0.5, color="red", label="Upper Bound")

    plt.title('RBF Kernel')
    # plt.legend(["Predictive Mean", "Finite Diff"])
    plt.legend(["Predictive Mean", "Finite Diff", "Lower Bound", "Upper Bound"])
    plt.show()

print('MADE IT!')