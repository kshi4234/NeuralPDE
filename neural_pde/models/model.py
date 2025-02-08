import torch
import torch.nn.functional as F
import gpytorch


from neural_pde.models.gaussian_process.standard_gp import StandardGaussianProcess, StandardGaussianProcessConfig
from neural_pde.models.gaussian_process.deep_kernel import DeepKernelGP, DeepKernelGPConfig, FeatureExtractor, FeatureExtractorConfig
from neural_pde.models.mlp.mlp import MLP, MLPConfig
from neural_pde.models.transformers.base_transformer import Transformer, TransformerConfig

# Neural Network Activation Functions
STR_TO_ACTIVATION = {
    "relu": F.relu,
    "gelu": F.gelu,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "softplus": F.softplus,
}

def get_model(args):
    if args.model == "mlp":
        config = MLPConfig(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            activation=STR_TO_ACTIVATION[args.activation],
        )
        return MLP(config)
    elif args.model == "transformer":
        config = TransformerConfig(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            n_embd=args.n_embd,
        )
        return Transformer(config)
    else:
        raise ValueError(f"Model \"{args.model}\" not supported")


# Gaussian Process Mean Functions
STR_TO_GP_MEAN = {
    "zero": gpytorch.means.ZeroMean,
    "constant": gpytorch.means.ConstantMean,
    "linear": gpytorch.means.LinearMean,
}

# Gaussian Process Covariance Functions
STR_TO_GP_COVAR = {
    "rbf": gpytorch.kernels.RBFKernel,
    "matern": gpytorch.kernels.MaternKernel,
    "linear": gpytorch.kernels.LinearKernel,
    "polynomial": gpytorch.kernels.PolynomialKernel,
}

# Gaussian Process Likelihood Functions
STR_TO_GP_LIKELIHOOD = {
    "gaussian": gpytorch.likelihoods.GaussianLikelihood,
}

# Deep Kernel Activation Functions
STR_TO_FEATURE_EXTRACTOR_ACTIVATION = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "softplus": torch.nn.Softplus,
}

def get_gp(args, x_train, y_train):

    if args.model == "gp":
        if args.gp_mean == "linear":
            mean_module = gpytorch.means.LinearMean(input_dim=args.input_dim)
        else:
            mean_module = STR_TO_GP_MEAN[args.gp_mean]()

        config = StandardGaussianProcessConfig(
            mean_module=mean_module,
            covar_module=STR_TO_GP_COVAR[args.gp_covar](),
        )

        likelihood = STR_TO_GP_LIKELIHOOD[args.gp_likelihood]()

        return StandardGaussianProcess(x_train, y_train, likelihood, config)
    
    elif args.model == "deep_kernel_gp":

        feature_extractor_config = FeatureExtractorConfig(
            input_dim=args.feature_extractor_input_dim,
            output_dim=args.feature_extractor_output_dim,
            hidden_dim=args.feature_extractor_hidden_dim,
            hidden_layers=args.feature_extractor_hidden_layers,
            activation=STR_TO_FEATURE_EXTRACTOR_ACTIVATION[args.activation],
        )

        feature_extractor = FeatureExtractor(feature_extractor_config)


        config = DeepKernelGPConfig(
            mean_module=STR_TO_GP_MEAN[args.gp_mean](),
            covar_module=STR_TO_GP_COVAR[args.gp_covar](),
        )
        likelihood = STR_TO_GP_LIKELIHOOD[args.gp_likelihood]()

        return DeepKernelGP(x_train, y_train, likelihood, feature_extractor, config)
    else:
        raise ValueError(f"Model \"{args.model}\" not supported")



