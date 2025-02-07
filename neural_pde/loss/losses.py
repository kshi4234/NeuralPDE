import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable

def get_loss(args):
    if args.problem == "square":
        return compute_loss_square
    elif args.problem == "rectangle":
        return rectangle_loss


def compute_laplacian(u, x, y):
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]        # [0] - selects gradient 
    du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    du2_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    du2_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0]

    laplacian = du2_dx2 + du2_dy2
    return laplacian

def boundary_loss(model, boundary_points, boundary_condition):
    """
    boundary_condition - tensor
    """
    outputs = model(boundary_points)
    return F.mse_loss(outputs, boundary_condition)

def compute_loss_square(
    model: nn.Module,
    u: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    boundary_conditions: Dict[str, float],
    gen_boundary_points_square: Callable
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

    bottom, top, left, right = gen_boundary_points_square(num_boundary_points)


    labels = torch.cat((bottom_boundary_label, top_boundary_label, left_boundary_label, right_boundary_label), dim=0)
    points = torch.cat((bottom, top, left, right), dim=0)

    boundary_mse = boundary_loss(model, points, labels)
    
    return laplacian_loss, boundary_mse


def rectangle_loss(model, u, x, y, boundary_conditions, domain_bounds, gen_boundary_points_rectangle):
    """
    Computes loss for PINN solving Laplace's equation on a rectangle
    """

    residual = compute_laplacian(u, x, y)
    laplacian_loss = torch.mean(residual**2)

    num_boundary_points = 25

    bottom, top, left, right = gen_boundary_points_rectangle(num_boundary_points=25, domain_bounds=domain_bounds)

    bottom_boundary_label = torch.tensor(boundary_conditions["bottom"], dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    top_boundary_label = torch.tensor(boundary_conditions["top"], dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    left_boundary_label = torch.tensor(boundary_conditions["left"], dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    right_boundary_label = torch.tensor(boundary_conditions["right"], dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)

    points = torch.cat((bottom, top, left, right), dim=0)
    labels = torch.cat((bottom_boundary_label, top_boundary_label, left_boundary_label, right_boundary_label), dim=0)
    boundary_mse = boundary_loss(model, points, labels)

    return laplacian_loss, boundary_mse

