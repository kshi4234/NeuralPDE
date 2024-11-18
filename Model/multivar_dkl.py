import gpytorch
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


# Define Model

train_x = torch.linspace(-7, 7, 15) 
train_y = torch.linspace(-7, 7, 15)
X, Y = torch.meshgrid(train_x, train_x, indexing="ij")
noise = torch.randn(size=X.flatten().size()) * 0.5

z = torch.sin(X.flatten()) + torch.cos(Y.flatten()) 
Z = z.reshape(X.shape)

xy_train = torch.cat((X.flatten().unsqueeze(1), Y.flatten().unsqueeze(1)), dim=-1)
z_train = Z.flatten()


# 3D Plot of data
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='viridis', edgecolor='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()



# GP Model

class FeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.add_module("linear1", torch.nn.Linear(2, 128))
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
model = DeepKernelGP(xy_train, z_train, likelihood, feature_extractor)

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
    output = model(xy_train)
    loss = -marginal_ll(output, z_train)
    loss.backward()
    optimizer.step()
    if (i + 1) % 10 == 0:
        print(f'Iter {i+1}/{100} - Loss: {loss.item()}, -Lengthscale: {model.covar_module.base_kernel.lengthscale.item()}, noise: {model.likelihood.noise.item()}')

# Evaluation
model.eval()
likelihood.eval()

test_x = torch.linspace(-7, 7, 15)
test_y = torch.linspace(-7, 7, 15)
test_X, test_Y = torch.meshgrid(test_x, test_y, indexing="ij")

with torch.no_grad():
    f_preds = model(torch.cat((test_X.flatten().unsqueeze(1), test_Y.flatten().unsqueeze(1)), dim=-1))
    f_samples = f_preds.sample(sample_shape=torch.Size([10]))

    fig = plt.figure(figsize=([10, 7]))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(X.flatten(), Y.flatten(), z_train, c="k", marker="*")

    # Sample Functions
    for i in range(f_samples.shape[0]):
        ax.plot_surface(
            test_X.numpy(),
            test_Y.numpy(), 
            f_samples[i].reshape(test_X.shape).numpy(),
            alpha=0.5,
            cmap="viridis",
            edgecolor="none"
        )
    
    lower, upper = f_preds.confidence_region()
    lower_surface = lower.reshape(test_X.shape).numpy() - 3.0
    upper_surface = upper.reshape(test_X.shape).numpy() + 3.0

    ax.plot_surface(test_X.numpy(), test_Y.numpy(), lower_surface, alpha=0.5, color="blue", label="Lower Bound")
    ax.plot_surface(test_X.numpy(), test_Y.numpy(), upper_surface, alpha=0.5, color="red", label="Upper Bound")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("Deep Kernel GP Multivariate Function Interpolation")
    plt.legend(["Observed Data", "Sample Functions", "Confidence Region"])
    plt.show()