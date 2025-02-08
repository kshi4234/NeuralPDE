import gpytorch


class StandardGaussianProcessConfig:
    mean_module: gpytorch.means.Mean = gpytorch.means.ConstantMean()
    covar_module: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel()


class StandardGaussianProcess(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, config: StandardGaussianProcessConfig):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = config.mean_module
        self.covar_module = gpytorch.kernels.ScaleKernel(config.covar_module)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        cov_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)