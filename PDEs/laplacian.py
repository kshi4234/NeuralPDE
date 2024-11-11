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

TODO: 
- Validation loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from NeuralPDE import MLP, DomainDataset
from plot_functions import plot_solution
from typing import Dict
import numpy as np

def generate_data(
    num_points: int = 10000,
    low: int = 0,
    high: int = 1
    ):
    """
    Requires grad must be true for gradient computation

    Returns training, validation datasets
    """

    x = torch.tensor(np.linspace(low, high, num_points).reshape(-1, 1).astype(np.float32), requires_grad=True)
    y = torch.tensor(np.linspace(low, high, num_points).reshape(-1, 1).astype(np.float32), requires_grad=True)


    domain_dataset = DomainDataset(x, y)
    domain_dataloader = DataLoader(domain_dataset, batch_size=64, shuffle=True)

    x_val = torch.tensor(np.linspace(low, high, num_points // 10).reshape(-1, 1).astype(np.float32), requires_grad=True)
    y_val = torch.tensor(np.linspace(low, high, num_points // 10).reshape(-1, 1).astype(np.float32), requires_grad=True)

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

def gen_boundary_points(num_boundary_points: int = 1000, low: int = 0, high: int = 1):
    """
    Generate boundary points on the domain [low, high] x [low, high]

    Returns concatenated boundary points: bottom, top, left, right each of which has 1000 points
    """
    x = torch.randint(low, high, (num_boundary_points,), dtype=torch.float32, requires_grad=True).unsqueeze(1)        # x and y are free on the boundary if the other is constrained to low / high
    y = torch.randint(low, high, (num_boundary_points,), dtype=torch.float32, requires_grad=True).unsqueeze(1)

    y_bottom = torch.tensor(float(low), dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    y_top = torch.tensor(float(high), dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    x_left = torch.tensor(float(low), dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    x_right = torch.tensor(float(high), dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)

    bottom = torch.cat((x, y_bottom), dim=-1)
    top = torch.cat((x, y_top), dim=-1)
    left = torch.cat((x_left, y), dim=-1)
    right = torch.cat((x_right, y), dim=-1)
    return bottom, top, left, right

def boundary_loss(model, boundary_points, boundary_condition):
    outputs = model(boundary_points)
    return F.mse_loss(outputs, boundary_condition)

def compute_loss(
    model: nn.Module,
    u: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    boundary_conditions: Dict[str, float]
    ):
    """
    Loss function for PINN solving Laplace's equation

    Boundary Loss: Generate boundary points and compare model output to the boundary condition
    - regularizing term from the boundary condition

    Args:
        boundary_conditions: Dictionary with keys "bottom", "top", "left", "right" and values as the boundary condition
    """

    # MSE of the Laplacian where the true value is 0
    residual = compute_laplacian(u, x, y)
    laplacian_loss = torch.mean(residual**2)     
    
    # Boundary Loss
    num_boundary_points = 25        

    # Generate 25 boundary points for each boundary
    bottom_boundary_label = torch.tensor(boundary_conditions["bottom"], dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    top_boundary_label = torch.tensor(boundary_conditions["top"], dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    left_boundary_label = torch.tensor(boundary_conditions["left"], dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    right_boundary_label = torch.tensor(boundary_conditions["right"], dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)

    bottom, top, left, right = gen_boundary_points(num_boundary_points)


    labels = torch.cat((bottom_boundary_label, top_boundary_label, left_boundary_label, right_boundary_label), dim=0)
    points = torch.cat((bottom, top, left, right), dim=0)

    boundary_mse = boundary_loss(model, points, labels)
    
    return laplacian_loss, boundary_mse




def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 1000,
    boundary_condition: Dict[str, float] = {"bottom": 3.0, "top": 3.0, "left": 3.0, "right": 3.0},
    alpha: float = 1.0
    ):
    laplace_losses = []
    boundary_losses = []


    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader):
            x, y = batch
            x, y = x.unsqueeze(1).float(), y.unsqueeze(1).float()

            optimizer.zero_grad()
            z = torch.cat((x, y), dim=-1)
            u = model(z)
            laplacian_loss, boundary_loss = compute_loss(model, u, x, y, boundary_condition)
            loss = boundary_loss + alpha * laplacian_loss 
            laplace_losses.append(laplacian_loss.item())
            boundary_losses.append(boundary_loss.item())
            loss.backward()
            optimizer.step()
        
        
        print(f"Epoch {epoch}")

    return laplace_losses, boundary_losses




if __name__ == "__main__":
    mlp = MLP(input_dim=2, hidden_dim=128, output_dim=1, depth=3, activation=F.tanh)
    train_dataloader, val_dataloader = generate_data(
        num_points=10000,
        low=0,
        high=1
    )


    boundary_condition = {"bottom": 3.0, "top": 3.0, "left": 3.0, "right": 3.0}



    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    laplace_losses, boundary_losses = train(mlp, train_dataloader, val_dataloader, optimizer, num_epochs=1)
    print(f"Initial Laplace Loss: {laplace_losses[0]}")
    print(f"Initial Boundary Loss: {boundary_losses[0]}")
    print(f"Laplace Loss: {laplace_losses[-1]}")
    print(f"Boundary Loss: {boundary_losses[-1]}")
    # print(f"Validation Loss: {val_losses[-1]}")
    plot_solution(mlp, low=0, high=1)



