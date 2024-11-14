"""
Rectangular domain with Dirichlet boundary conditions

M_b training method: L(u) = g(u(x, y))
"""


import os
import sys

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(module_path)

from utils.data_utils import (
    gen_interior_points_rectangle, data_square, data_loader_square
)

from utils.laplacian_utils import (
    train, rectangle_loss, train_no_batches_rectangle
)

from models.NeuralPDE import (
    ADF_MLP
)       

from utils.plot_utils import (
    plot_solution
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from tqdm import tqdm
import matplotlib.pyplot as plt


# Custom Distance Function

def ADF_rectangle(z, domain_bounds: Dict[str, float]):
    """
    Computes the minimum distance from a point to the boundary of a rectangle
    """
    x_min = float(domain_bounds["left"])
    x_max = float(domain_bounds["right"])
    y_min = float(domain_bounds["bottom"])
    y_max = float(domain_bounds["top"])

    x = z[:, 0]
    y = z[:, 1]

    x_min_dist = torch.min(torch.stack([x - x_min, x_max - x]).permute(1, 0), dim=-1).values
    y_min_dist = torch.min(torch.stack([y - y_min, y_max - y]).permute(1, 0), dim=-1).values
        
    min_dist = torch.min(torch.stack([x_min_dist, y_min_dist]).permute(1, 0), dim=-1).values

    # print(z)
    # print(min_dist)

    return min_dist


def get_boundary_values(z, boundary_condition: Dict[str, float], domain_bounds: Dict[str, float]):
    """
    Gets the boundary values for a point

    If the boundary is composed for 4 segments and (x, y) happens to lie 
    on one of the segments, then the boundary value is the value of the segment

    TODO - figure out what to do with points not on the boundary
    """

    x = z[:, 0]
    y = z[:, 1]

    boundary_values = torch.zeros_like(x)

    boundary_values = torch.where(y == float(domain_bounds["bottom"]), torch.tensor(boundary_condition["bottom"]), boundary_values)
    boundary_values = torch.where(y == float(domain_bounds["top"]), torch.tensor(boundary_condition["top"]), boundary_values)
    boundary_values = torch.where(x == float(domain_bounds["left"]), torch.tensor(boundary_condition["left"]), boundary_values)
    boundary_values = torch.where(x == float(domain_bounds["right"]), torch.tensor(boundary_condition["right"]), boundary_values)

    return boundary_values


# Testing Code for ADF_rectangle
# z = torch.tensor([[0.4, 0.5], [0.75, 0.6], [0.53, 0.50], [0.115, 0.114], [0.0, 0.0]])
# domain_bounds = {"bottom": 0, "left": 0, "top": 1, "right": 1}
# print(ADF_rectangle(z, domain_bounds))
    


        



if __name__ == "__main__":
    boundary_condition = {"bottom": 1.0, "top": 0, "left": 1.0, "right": 0}     # this is the boundary condition
    domain_bounds = {"bottom": 0, "left": 0, "top": 1, "right": 1}            # this is the domain bounds    

    ADF_PINN = ADF_MLP(
        input_dim=2, 
        hidden_dim=100, 
        output_dim=1, 
        depth=3, 
        activation=F.relu, 
        boundary_condition=boundary_condition,
        domain_bounds=domain_bounds, 
        adf=ADF_rectangle,
        boundary_value_func=get_boundary_values
    )

    optimizer = torch.optim.AdamW(ADF_PINN.parameters(), lr=0.01)

    x_train, y_train = gen_interior_points_rectangle(5, domain_bounds)

    train_no_batches_rectangle(
        ADF_PINN, 
        x_train, 
        y_train, 
        optimizer, 
        num_epochs=100, 
        boundary_condition=boundary_condition, 
        domain_bounds=domain_bounds,
        boundary_func=rectangle_loss,
        alpha=1.0,
        shuffle=True
    )

    plot_solution(ADF_PINN, domain_bounds, id=-1)

    




