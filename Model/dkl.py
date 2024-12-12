import gpytorch
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# Define Model

train_x = torch.linspace(0, 2.0, 10) 
noise = torch.randn(10) * 0.3
train_y = 2.0 * train_x + noise

train_x_std = (train_x - train_x.mean()) / train_x.std()
train_y_std = (train_y - train_y.mean()) / train_y.std()


plt.scatter(train_x_std, train_y_std)
plt.show()

class FeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.add_module("linear1", torch.nn.Linear(1, 128))
        self.add_module("relu1", torch.nn.Tanh())
        self.add_module("linear2", torch.nn.Linear(128, 128))
        self.add_module("relu2", torch.nn.Tanh())
        self.add_module("linear3", torch.nn.Linear(128, 1))
    
class DeepKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
    
    def forward(self, x):
        proj_x = self.feature_extractor(x)
        proj_x = self.scale_to_bounds(proj_x)
        
        mean_x = self.mean_module(proj_x)
        covar_x = self.covar_module(proj_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()

feature_extractor = FeatureExtractor()
model = DeepKernelGP(train_x_std, train_y_std, likelihood, feature_extractor)
hyper_params = {
    "likelihood.noise_covar.noise": 0.01,
    "covar_module.base_kernel.lengthscale": 0.4,
    "covar_module.outputscale": 1.0,
}
model.likelihood.noise_covar.raw_noise.requires_grad = False

model.initialize(**hyper_params)

model.train()
likelihood.train()
marginal_ll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
loss_list = []
for i in tqdm(range(100)):
    optimizer.zero_grad()
    output = model(train_x_std)
    loss = -marginal_ll(output, train_y_std)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())

plt.plot(loss_list)
plt.show()

# Evaluation

model.eval()
likelihood.eval()

test_x = torch.linspace(0, 2.0, 15) # scale further out
test_x_std = (test_x - train_x.mean()) / train_x.std()
test_y = 2.0 * test_x

with torch.no_grad():
    f_preds = model(test_x_std)
    f_samples = f_preds.sample(sample_shape=torch.Size([5]))   # Generate 10 Samples

    plt.figure(figsize=(10, 6))
    plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Observed Data', color='black')
    plt.plot(test_x.numpy(), test_y.numpy(), 'k-', label='True Function', color='red')
    # Sampled Functions
    for i in range(f_samples.shape[0]):
        plt.plot(test_x.numpy(), (f_samples[i] * train_y.std() + train_y.mean()).numpy(), alpha=0.8, label=f"Sample {i}")

    lower, upper = f_preds.confidence_region()
    plt.fill_between(test_x.numpy(), (lower * train_y.std() + train_y.mean()).numpy(), (upper * train_y.std() + train_y.mean()).numpy(), color="mediumpurple", alpha=0.5, label="Confidence Region")
    plt.title("Deep Kernel Function Interpolation")
    plt.legend()
    plt.show()

for name, param in model.named_parameters():
    if param.numel() == 1:
        print(f"{name}: {param.item()}")


