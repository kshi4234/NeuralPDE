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
Output: u(x, y)

Loss function: 
MSE + Boundary loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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


class DomainDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def generate_data(
    num_points: int = 10000
    ):
    """
    Requires grad must be true for gradient computation

    Returns training, validation datasets
    """

    x = torch.rand(num_points, requires_grad=True)              # generate num_points random point in Domain: [0, 1] x [0, 1]
    y = torch.rand(num_points, requires_grad=True)

    domain_dataset = DomainDataset(x, y)
    domain_dataloader = DataLoader(domain_dataset, batch_size=64, shuffle=True)

    x_val = torch.rand(num_points // 10, requires_grad=True)
    y_val = torch.rand(num_points // 10, requires_grad=True)

    val_dataset = DomainDataset(x_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    return domain_dataloader, val_dataloader


# Interior Functions

def compute_laplacian(u, x, y):
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]        # [0] - selects gradient 
    du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    du2_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    du2_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0]

    laplacian = du2_dx2 + du2_dy2
    return laplacian




# Boundary Functions

def gen_boundary_points(num_boundary_points: int = 1000):
    x = torch.rand(num_boundary_points, requires_grad=True).unsqueeze(1)        # x and y are free on the boundary if the other is constrained to 0 or 1
    y = torch.rand(num_boundary_points, requires_grad=True).unsqueeze(1)

    y_zeros = torch.zeros_like(x, requires_grad=True)
    y_ones = torch.ones_like(x, requires_grad=True)
    x_zeros = torch.zeros_like(y, requires_grad=True)
    x_ones = torch.ones_like(y, requires_grad=True)


    bottom = torch.cat((x, y_zeros), dim=-1)
    top = torch.cat((x, y_ones), dim=-1)
    left = torch.cat((x_zeros, y), dim=-1)
    right = torch.cat((x_ones, y), dim=-1)
    return torch.cat((bottom, top, left, right), dim=0)

def boundary_loss(model, boundary_points, boundary_condition):
    outputs = model(boundary_points)
    return F.mse_loss(outputs, boundary_condition)


def compute_loss(
    model: nn.Module,
    u: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    boundary_condition: int
    ):
    """
    Loss function for PINN solving Laplace's equation

    Boundary Loss: Generate boundary points and compare model output to the boundary condition
    - regularizing term from the boundary condition
    """

    # MSE of the Laplacian where the true value is 0
    residual = compute_laplacian(u, x, y)
    interior_mse = torch.mean(residual**2)     
    
    # Boundary Loss
    num_boundary_points = 25        
    boundary_values = torch.Tensor.expand(torch.tensor(boundary_condition), size=(4 * num_boundary_points, 1))
    boundary_points = gen_boundary_points(num_boundary_points)
    boundary_mse = boundary_loss(model, boundary_points, boundary_values)

    return interior_mse + boundary_mse




def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 1000,
    boundary_condition: float = 0.0
    ):

    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader):
            x, y = batch
            x, y = x.unsqueeze(1).float(), y.unsqueeze(1).float()

            optimizer.zero_grad()
            z = torch.cat((x, y), dim=-1)
            u = model(z)
            loss = compute_loss(model, u, x, y, boundary_condition)
            loss.backward()
            optimizer.step()
        
        print("EPOCH FINISHED")
        model.eval()
        for batch in tqdm(val_dataloader):
            x, y = batch
            x, y = x.unsqueeze(1).float(), y.unsqueeze(1).float()
            z = torch.cat((x, y), dim=-1)
            u = model(z)
            loss = compute_loss(model, u, x, y, boundary_condition) 
            val_losses.append(loss.item())

        val_losses[epoch] /= len(val_dataloader)
        print(f"Epoch {epoch}, Validation Loss: {val_losses[epoch]}")

    return val_losses



if __name__ == "__main__":
    mlp = MLP(input_dim=2, hidden_dim=40, output_dim=1, depth=1, activation=F.tanh)
    train_dataloader, val_dataloader = generate_data(
        num_points=10000
    )

    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    val_losses = train(mlp, train_dataloader, val_dataloader, optimizer, num_epochs=1000)



