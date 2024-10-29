"""
Laplace's equation: Delta f = 0
Delta is the Laplace operator: sum of second derivatives
= d2f/dx2 + d2f/dy2

Question: How do we solve Delta f = 0 using a PINN?

Modeling: 
We have some domain and some function f which tells us the temperature at each point in the domain.
We want to learn f.
We have some domain D and boundary B.
If (x, y) is in B, then we know f - its equal to some known function g or some constant c.

Inputs: (x, y)
Outputs: u(x, y)


Loss function: 
MSE + Boundary loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, activation=F.tanh):
        """
        Defines basic MLP architecture
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


def generate_data(
    num_points: int = 1000, 
    domain: tuple[float, float] = (0, 1), 
    dim_domain: int = 2, 
    boundary_conditions: dict = None,
    ):

    x = torch.rand(num_points)              # generate num_points random point in Domain: [0, 1] x [0, 1]
    y = torch.rand(num_points)


def compute_laplacian(model, x, y):
    u = model(x, y)
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    laplacian = du_dx + du_dy
    return laplacian

# Boundary Functions

def gen_boundary_points(num_boundary_points: int = 1000):
    x = torch.rand(num_boundary_points).unsqueeze(1)        # x and y are free on the boundary if the other is constrained to 0 or 1
    y = torch.rand(num_boundary_points).unsqueeze(1)

    y_zeros = torch.zeros_like(x)
    y_ones = torch.ones_like(x)
    x_zeros = torch.zeros_like(y)
    x_ones = torch.ones_like(y)


    bottom = torch.cat((x, y_zeros), dim=-1)
    top = torch.cat((x, y_ones), dim=-1)
    left = torch.cat((x_zeros, y), dim=-1)
    right = torch.cat((x_ones, y), dim=-1)
    return torch.cat((bottom, top, left, right), dim=0)

def boundary_loss(model, boundary_points, boundary_condition):
    outputs = model(boundary_points)
    return F.mse_loss(outputs, boundary_condition)


def loss(model, x, y, boundary_condition):
    """
    Loss function for PINN solving Laplace's equation

    Boundary Loss: Generate boundary points and compare model output to the boundary condition
    - regularizing term from the boundary condition
    """
    laplacian = compute_laplacian(model, x, y)
    laplacian_mse = F.mse_loss(laplacian, torch.zeros_like(laplacian))
    
    # Boundary Loss
    num_boundary_points = 25        
    boundary_values = torch.Tensor.expand(torch.tensor(boundary_condition), size=(4 * num_boundary_points, 1))
    boundary_points = gen_boundary_points(num_boundary_points)
    boundary_mse = boundary_loss(model, boundary_points, boundary_values)

    return boundary_mse + laplacian_mse



if __name__ == "__main__":
    mlp = MLP(input_dim=2, hidden_dim=10, output_dim=1, depth=1, activation=F.tanh)



