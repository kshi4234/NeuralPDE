"""
Essentially implementation of the example from the GPyTorch tutorial
https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
"""

import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np




x = torch.linspace(0, 1, 10)
noise = torch.randn(10) * 0.3
y = 3.0 * x + noise

class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)
    
# Setting the likelihood noise so low --> conditions posterior to pass through training points
noise_constraint = gpytorch.constraints.GreaterThan(1e-9)
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
likelihood.noise = torch.tensor(1e-6)
likelihood.noise_covar.raw_noise.requires_grad = False

model = GP(x, y, likelihood)

# Training
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
marginal_ll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = -marginal_ll(output, y)
    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
        print(f'Iter {i+1}/{100} - Loss: {loss.item()}, -Lengthscale: {model.covar_module.base_kernel.lengthscale.item()}, noise: {model.likelihood.noise.item()}')



# Predictions
model.eval()
likelihood.eval()

test_x = torch.linspace(0, 1, 100)
with torch.no_grad():
    f_preds = model(test_x)     # This generates a MVG
    f_samples = f_preds.sample(sample_shape=torch.Size([10]))

# Plotting
with torch.no_grad():
    plt.figure(figsize=(10, 6))
    # observed data
    plt.plot(x.numpy(), y.numpy(), 'k*', label='Observed Data')

    # sampled functions
    for i in range(f_samples.shape[0]):
        plt.plot(test_x.numpy(), f_samples[i].numpy(), linewidth=1.0, alpha=0.8)    

    # confidence region
    lower, upper = f_preds.confidence_region()
    plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), color='mediumpurple', alpha=0.5)
    plt.ylim([0.0, 3.5])
    plt.legend(['Observed Data', 'Sample Functions', 'Confidence Region'])
    plt.title("GP Function Interpolation")
    plt.show()


    # # Plot observed data
    # ax.plot(x.numpy(), y.numpy(), 'k*', label='Observed Data')

    # # Plot sampled functions
    # for i in range(f_samples.shape[0]):
    #     ax.plot(test_x.numpy(), f_samples[i].numpy(), linewidth=1.0, alpha=1.0)    

    # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), color='mediumpurple', alpha=0.5)
    # ax.set_ylim([-2, 2.0])
    # ax.legend(['Observed Data', 'Sample Functions'])
    # plt.show()