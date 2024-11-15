#TODO: THIS IS ONLY SUITED FOR LINEAR REGRESSION (as of now). WILL NEED TO BE CHANGED FOR PDES! Also try for higher dimensional inputs; only has been tested on 1-dimensional inputs
#TODO: Need to separate out some things to make it more modular

# Create and train the model
# Needs to import from loss_utils.py to get the loss functions
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

from GP import(
    GP,
)
from utils.loss_utils import(
    Generic_Loss,
    mse_loss,
)
from Regression.regression_data import(
    gen_data
)

# Currently testing linear regression
# Using mll, take negative to minimize
# TODO: Delete batch_size parameter, no batching for Gaussian process
def train(model, train_data, val_data, optimizer, loss_fn, epochs=20):
    iterator = tqdm(range(epochs))
    for epoch in iterator:
        x, y = train_data
        output = model(x)
        loss = -loss_fn(output, y)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss = loss.detach().clone()
        
        # val_x, val_y = val_data
        # output = model(val_x)
        # loss = -loss_fn(output, val_y)
        # val_loss = loss.mean().detach().clone()    # TODO: Print val_loss (first make sure training works)
        iterator.set_description('EPOCH %d - TRAIN LOSS = %d' % (epoch, epoch_loss))
    return

def test(model, likelihood, test_data, loss_fn, batch_size=1):
    model.eval()
    likelihood.eval()
    test_loss = 0
    with torch.no_grad():
        x, y = test_data
        output = model(x)
        preds = likelihood(output)
        # print('PREDS:', preds)
        # loss = -loss_fn(preds, y)
        # test_loss = loss.detach().clone()
        # print('TEST LOSS = %d' % (test_loss))
    return preds

# Generate Data
train_x, train_y, val_x, val_y = gen_data(datapoints=100, train=True)

# Set kernel and likelihood
mean = gpytorch.means.ConstantMean()
covar = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
distribution = gpytorch.distributions.MultivariateNormal
# Define model and log marginal likelihood (loss function for optimization)
model = GP(train_x=train_x, train_y=train_y, mean=mean, covar=covar, likelihood=likelihood, distribution=distribution)
if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


optimizer = torch.optim.Adam([
    {'params': model.transform.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.1)

train(model=model,
      train_data=(train_x, train_y),
      val_data=(val_x, val_y),
      optimizer=optimizer,
      loss_fn=mll,
      epochs=100,
      )

test_x, test_y = gen_data(datapoints=100, train=False)

prediction = test(model=model,
                  likelihood=likelihood,
                  test_data=(test_x, test_y),
                  loss_fn=mse_loss)


f, ax = plt.subplots(1, 1, figsize=(4, 3))
lower, upper = prediction.confidence_region()
# Plot training data as black stars
ax.plot(train_x.squeeze().numpy(), train_y.numpy(), 'k*')
# Plot predictive means as blue line
ax.plot(test_x.squeeze().numpy(), prediction.mean.numpy(), 'b')
# Shade between the lower and upper confidence bounds
ax.fill_between(test_x.squeeze().numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
ax.set_ylim([0, 30])
ax.legend(['Observed Data', 'Mean', 'Confidence'])
plt.show()