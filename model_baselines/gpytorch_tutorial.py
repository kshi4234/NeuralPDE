"""
Essentially implementation of the example from the GPyTorch tutorial
https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
"""

import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np

from neural_pde.gaussian_process.standard_gp import StandardGaussianProcess, StandardGaussianProcessConfig


x = torch.linspace(0, 1, 10)
noise = torch.randn(10) * 0.1
y = 3.0 * x + noise

plt.scatter(x, y)
plt.title("Training Data")
plt.show()

config = StandardGaussianProcessConfig()    

# Setting the likelihood noise so low --> conditions posterior to pass through training points
noise_constraint = gpytorch.constraints.GreaterThan(1e-9)
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
likelihood.noise = torch.tensor(1e-5)
likelihood.noise_covar.raw_noise.requires_grad = False

model = StandardGaussianProcess(x, y, likelihood, config)

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
    f_preds = model(test_x)     
    f_samples = f_preds.sample(sample_shape=torch.Size([10]))

# Plotting
with torch.no_grad():
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y.numpy(), 'k*', label='Observed Data')

    for i in range(f_samples.shape[0]):
        plt.plot(test_x.numpy(), f_samples[i].numpy(), linewidth=1.0, alpha=0.8)    

    lower, upper = f_preds.confidence_region()
    plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), color='mediumpurple', alpha=0.5)
    plt.legend(['Observed Data', 'Sample Functions', 'Confidence Region'])
    plt.title("GP Function Interpolation")
    plt.show()


