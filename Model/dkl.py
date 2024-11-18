import gpytorch
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# Define Model

train_x = torch.linspace(0, 2.0, 10) 
noise = torch.randn(10) * 0.5
train_y = 2.0 * train_x + noise

plt.scatter(train_x, train_y)
plt.show()


class FeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.add_module("linear1", torch.nn.Linear(1, 128))
        self.add_module("relu1", torch.nn.ReLU())
        self.add_module("linear2", torch.nn.Linear(128, 128))
        self.add_module("relu2", torch.nn.ReLU())
        self.add_module("linear3", torch.nn.Linear(128, 1))
    
        



class DeepKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
    
    def forward(self, x):
        proj_x = self.feature_extractor(x)
        proj_x = self.scale_to_bounds(proj_x)
        mean_x = self.mean_module(proj_x)
        covar_x = self.covar_module(proj_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

noise_constraint = gpytorch.constraints.GreaterThan(1e-9)
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
likelihood.noise = torch.tensor(1e-4)
likelihood.noise_covar.raw_noise.requires_grad = False  # Don't train noise

feature_extractor = FeatureExtractor()
model = DeepKernelGP(train_x, train_y, likelihood, feature_extractor)

model.train()
likelihood.train()
marginal_ll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

optimizer = torch.optim.Adam([
    {"params": model.feature_extractor.parameters()},
    {"params": model.covar_module.parameters()},
    {"params": model.mean_module.parameters()},
    {"params": model.likelihood.parameters()}
], lr=0.01)


# Training
for i in tqdm(range(100)):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -marginal_ll(output, train_y)
    loss.backward()
    optimizer.step()
    if (i + 1) % 10 == 0:
        print(f'Iter {i+1}/{100} - Loss: {loss.item()}, -Lengthscale: {model.covar_module.base_kernel.lengthscale.item()}, noise: {model.likelihood.noise.item()}')

# Evaluation


model.eval()
likelihood.eval()

test_x = torch.linspace(0, 2.0, 30)

with torch.no_grad():
    f_preds = model(test_x)
    f_samples = f_preds.sample(sample_shape=torch.Size([15]))   # Generate 10 Samples

    plt.figure(figsize=(10, 6))
    plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Observed Data')

    # Sampled Functions
    for i in range(f_samples.shape[0]):
        plt.plot(test_x.numpy(), f_samples[i].numpy(), linewidth=1.0, alpha=0.8)

    
    lower, upper = f_preds.confidence_region()
    plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), color="mediumpurple", alpha=0.5)
    plt.legend(["Observed Data", "Sample Functions", "Confidence Region"])
    plt.title("Deep Kernel Function Interpolation")
    plt.show()
