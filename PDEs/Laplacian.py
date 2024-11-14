# Will generate the Laplacian data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

from utils.data_utils import (
    generate_data, generate_data_loader
)

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

    x = torch.tensor(np.linspace(low, high, num_boundary_points).reshape(-1, 1).astype(np.float32))    # x and y are free on the boundary if the other is constrained to low / high
    y = torch.tensor(np.linspace(low, high, num_boundary_points).reshape(-1, 1).astype(np.float32))

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

