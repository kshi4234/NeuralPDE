import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

"""
Dirichlet Boundary Condition: u(x, y) = f(x, y) for (x, y) on the boundary B
"""


# Models

class VanillaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, activation=F.tanh):
        """
        Defines basic VanillaMLP architecture

        `Weighting Scheme`: for Dirichlet boundary conditions, weight error on interior + error on boundary
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.activation = activation

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, z):
        z = self.input_layer(z)
        z = self.activation(z)

        for layer in self.hidden_layers:
            z = layer(z)
            z = self.activation(z)
        
        z = self.output_layer(z) 
        return z

    def test_forward(self):
        x = torch.randn(20, self.input_dim)
        y = self.forward(x)
        return y



class ImpositionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, activation, weak_distance_f):
        """
        https://www.sciencedirect.com/science/article/pii/S2405844023060280#coi0001
        Method M_B

        Gets boundary condition 'for free'
        """

        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.activation = activation
        self.weak_distance_f = weak_distance_f
    
    def forward(self, z):
        z1 = self.fc1(z)

        for layer in self.hidden_layers:
            z1 = layer(z1)
            z1 = self.activation(z1)
        
        z1 = self.output_layer(z1)
        return self.weak_distance_f(z1)
    



















# Dataset Classes

class DomainDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]