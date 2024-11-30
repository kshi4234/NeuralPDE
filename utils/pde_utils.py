"""
Generic util file for PDEs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Callable, Dict
from utils.data_utils import gen_boundary_points_rectangle


def compute_boundary_loss_rectangle(model, domain_bounds, boundary_condition, points_per_boundary_segment):
    """
    Compute the boundary loss for a rectangle
    """

    # Input boundary points
    bottom, top, left, right = gen_boundary_points_rectangle(points_per_boundary_segment, domain_bounds)

    # Label boundary points
    bottom_label = torch.tensor(boundary_condition["bottom"], dtype=torch.float32, requires_grad=True).repeat(points_per_boundary_segment).unsqueeze(1)
    top_label = torch.tensor(boundary_condition["top"], dtype=torch.float32, requires_grad=True).repeat(points_per_boundary_segment).unsqueeze(1)
    left_label = torch.tensor(boundary_condition["left"], dtype=torch.float32, requires_grad=True).repeat(points_per_boundary_segment).unsqueeze(1)
    right_label = torch.tensor(boundary_condition["right"], dtype=torch.float32, requires_grad=True).repeat(points_per_boundary_segment).unsqueeze(1)

    labels = torch.cat((bottom_label, top_label, left_label, right_label), dim=0)
    points = torch.cat((bottom, top, left, right), dim=0)

    output = model(points)  
    boundary_mse = F.mse_loss(output, labels)

    return boundary_mse


def compute_boundary_loss_circle(model, radius, center, boundary_func, point_count):
    """
    Compute the boundary loss for a circle
    """

    t = torch.linspace(0, 2*np.pi, point_count)
    









def train_no_batches(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    boundary_condition: Dict[str, float],
    domain_bounds: Dict[str, float],
    num_epochs: int = 1000,
    shuffle: bool = False,
):
    """
    Loss_fn should take in:
    - model: to compute the output on the boundary
    - u: model output on the interior
    - x, y: input points for nth order derivatives
    - boundary_condition: boundary condition for the problem
    - domain_bounds: domain bounds for the problem
    """
    
    train_x, train_y = train_x.float(), train_y.float()
    interior_loss = []
    boundary_loss = []


    for _ in enumerate(tqdm(range(num_epochs))):
        model.train()
        optimizer.zero_grad()
        z = torch.cat((train_x, train_y), dim=-1)
        if shuffle:
            idx = torch.randperm(z.shape[0])
            z = z[idx].view(z.size())

        u = model(z)
        interior_err, boundary_err = loss_fn(model, u, train_x, train_y, boundary_condition, domain_bounds)
        loss: torch.Tensor = interior_err + boundary_err
        
        interior_loss.append(interior_err.item())
        boundary_loss.append(boundary_err.item())

        loss.backward()
        optimizer.step()

    return interior_loss, boundary_loss


