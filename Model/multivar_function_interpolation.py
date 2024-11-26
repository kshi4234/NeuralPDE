"""
Multivariate function interpolation with Gaussian Processes
"""

import torch
import gpytorch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np



# sample from [-7, 7] x [-7, 7]
x = torch.linspace(-7, 7, 100)
y = torch.linspace(-7, 7, 100)
X, Y = torch.meshgrid(x, y, indexing="ij")

z = torch.sin(X.flatten()) + torch.cos(Y.flatten())
Z = z.reshape(X.shape)

# 3D Plot of the data
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='viridis', edgecolor='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()



# GP Code

x_train = torch.linspace(-7, 7, 10)
y_train = torch.linspace(-7, 7, 10)
X_train, Y_train = torch.meshgrid(x_train, y_train, indexing="ij")
Z_train = torch.sin(X_train) + torch.cos(Y_train)

x_train_flat = X_train.flatten()
y_train_flat = Y_train.flatten()
z_train_flat = Z_train.flatten()

xy_train = torch.cat((x_train_flat.unsqueeze(1), y_train_flat.unsqueeze(1)), dim=-1)


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov_x)

# Set likelihood noise to be low --> forces posterior to pass through training points
noise_constraint = gpytorch.constraints.GreaterThan(1e-9)
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
likelihood.noise = torch.tensor(1e-6)
likelihood.noise_covar.raw_noise.requires_grad = False  # Does this need to be here?

model = GPModel(xy_train, z_train_flat, likelihood)

# Training
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
marginal_ll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    optimizer.zero_grad()
    output = model(xy_train)
    loss = -marginal_ll(output, z_train_flat)
    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
        print(f'Iter {i+1}/{100} - Loss: {loss.item()}, -Lengthscale: {model.covar_module.base_kernel.lengthscale.item()}, noise: {model.likelihood.noise.item()}')


# Predictions
model.eval()
likelihood.eval()

test_x = torch.linspace(-7, 7, 15)
test_y = torch.linspace(-7, 7, 15)

test_X, test_Y = torch.meshgrid(test_x, test_y, indexing="ij")

test_X_flat = test_X.flatten()
test_Y_flat = test_Y.flatten()

test_xy = torch.cat((test_X_flat.unsqueeze(1), test_Y_flat.unsqueeze(1)), dim=-1)

with torch.no_grad():
    f_preds = model(test_xy)     # Generates MVG
    f_samples = f_preds.sample(sample_shape=torch.Size([10]))


# Plotting
with torch.no_grad():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")

    # Observed data points
    ax.scatter(x_train_flat, y_train_flat, z_train_flat, c="k", marker="*", label="Observed Data")


    # Sample functions
    for i in range(f_samples.shape[0]):
        ax.plot_surface(
            test_X.numpy(),
            test_Y.numpy(),
            f_samples[i].reshape(test_X.shape).numpy(),
            alpha=0.4,
            cmap='viridis',
            edgecolor='none',
        )

        # f_samples[i].shape: [225]
        # test_X.shape: [15, 15]

    # Confidence region
    lower, upper = f_preds.confidence_region()
    lower_surface = lower.reshape(test_X.shape).numpy() - 3.0
    upper_surface = upper.reshape(test_X.shape).numpy() + 3.0

    ax.plot_surface(test_X.numpy(), test_Y.numpy(), lower_surface, alpha=0.5, color="blue", label="Lower Bound")
    ax.plot_surface(test_X.numpy(), test_Y.numpy(), upper_surface, alpha=0.5, color="red", label="Upper Bound")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("GP Multivariate Function Interpolation")
    plt.legend(["Observed Data", "Sample Functions", "Confidence Region"])
    plt.show()


