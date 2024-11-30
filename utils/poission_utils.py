"""
Poisson's equation: Delta f = phi


Interior Loss:
L_interior(f) = ||Delta f - phi||^2

Boundary Loss:
L_boundary(f) = ||f - g||^2

Question:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable

def interior_loss(
    u: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    phi: Callable
    ):

    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    du2_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    du2_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0]

    laplacian = du2_dx2 + du2_dy2

    return F.mse_loss(laplacian, phi(x, y))

def poisson_loss(
    model: nn.Module,
    u: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    phi: Callable,
    domain_bounds: Dict[str, float]
    ):

    # MSE of Laplacian and phi
    interior_error = interior_loss(u, x, y, phi)

    # model output on the boundary

    b

