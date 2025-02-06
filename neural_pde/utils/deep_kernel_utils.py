# Implements the deep network that serves as the non-linear transform.
# The outputs of this network will be used as the input to the Gaussian Process
# We use pytorch to construct the multilayer network.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Might want to try different activation functions for differentiability

class Deep_Transform(nn.Sequential):
    def __init__(self, input_dim, hidden_dim=1000, output_dim=2):
        super(Deep_Transform, self).__init__()
        self.add_module('linear1', torch.nn.Linear(input_dim, hidden_dim))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(hidden_dim, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, output_dim))
        