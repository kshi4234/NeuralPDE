import torch
import gpytorch
from dataclasses import dataclass

@dataclass
class FeatureExtractorConfig:
    hidden_layers: int = 3
    hidden_dim: int = 128
    input_dim: int = 1
    output_dim: int = 1
    activation: str = "Tanh"

@dataclass
class DeepKernelGPConfig:
    mean_module: gpytorch.means.Mean = gpytorch.means.ZeroMean()
    covar_module: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel()

class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, config: FeatureExtractorConfig):
        super().__init__()
        self.add_module("linear1", torch.nn.Linear(config.input_dim, config.hidden_dim))
        self.add_module(config.activation, getattr(torch.nn, config.activation)())

        for i in range(config.hidden_layers):
            self.add_module(f"linear{i+2}", torch.nn.Linear(config.hidden_dim, config.hidden_dim))
            self.add_module(config.activation, getattr(torch.nn, config.activation)())
            config.input_dim = config.hidden_dim

        self.add_module(f"linear{config.hidden_layers + 2}", torch.nn.Linear(config.hidden_dim, config.output_dim))



class DeepKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, config: DeepKernelGPConfig):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = config.mean_module
        self.covar_module = gpytorch.kernels.ScaleKernel(config.covar_module)
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
    
    def forward(self, x):
        proj_x = self.feature_extractor(x)
        proj_x = self.scale_to_bounds(proj_x)
        
        mean_x = self.mean_module(proj_x)
        covar_x = self.covar_module(proj_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



    