"""
Data Util files - generates interior and boundary points, data loaders
"""

import os, sys

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(module_path)

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Dict


# Data Loaders

class DomainDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def data_loader_square(
    num_points: int = 10000,
    low: int = 0,
    high: int = 1
):
    """
    Requires grad must be true for gradient computation

    Returns training, validation datasets. 
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

def data_square(
    num_points: int = 10000,
    low: int = 0,
    high: int = 1
):
    """
    Generates training, validation data. Returns tensors instead of datasets. 

    If the dataset is small enough, we don't need to use batching. 
    """

    x = (high - low) * torch.rand(num_points) + low        # Maps Unif[0, 1] -> Unif[low, high]
    y = (high - low) * torch.rand(num_points) + low
    x = x.unsqueeze(1).float()
    y = y.unsqueeze(1).float()
    x.requires_grad = True
    y.requires_grad = True


    x_val = (high - low) * torch.rand(num_points) + low        # Maps Unif[0, 1] -> Unif[low, high]
    y_val = (high - low) * torch.rand(num_points) + low
    x_val = x_val.unsqueeze(1).float()
    y_val = y_val.unsqueeze(1).float()
    x_val.requires_grad = True
    y_val.requires_grad = True

    return x, y, x_val, y_val

# Can I get rid of this?

def gen_boundary_points_square(num_boundary_points: int = 1000, low: int = 0, high: int = 1):
    """
    Generate boundary points on the domain [low, high] x [low, high]

    Returns concatenated boundary points: bottom, top, left, right each of which has 1000 points
    """

    x = (high - low) * torch.rand(num_boundary_points) + low        # Maps Unif[0, 1] -> Unif[low, high]
    y = (high - low) * torch.rand(num_boundary_points) + low
    x = x.unsqueeze(1).float()
    y = y.unsqueeze(1).float()
    
    # x = torch.tensor(np.linspace(low, high, num_boundary_points).reshape(-1, 1).astype(np.float32))    # x and y are free on the boundary if the other is constrained to low / high
    # y = torch.tensor(np.linspace(low, high, num_boundary_points).reshape(-1, 1).astype(np.float32))

    y_bottom = torch.tensor(float(low), dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    y_top = torch.tensor(float(high), dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    x_left = torch.tensor(float(low), dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    x_right = torch.tensor(float(high), dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)

    bottom = torch.cat((x, y_bottom), dim=-1)
    top = torch.cat((x, y_top), dim=-1)
    left = torch.cat((x_left, y), dim=-1)
    right = torch.cat((x_right, y), dim=-1)
    return bottom, top, left, right



# Utilized in experiments

def gen_interior_points_rectangle(
    num_interior_points: int = 1000, 
    domain_bounds: Dict[str, float] = {"bottom": 0, "left": 0, "top": 4, "right": 2}
):
    """
    Generates interior points on the domain [low, high] x [low, high]

    domain_bounds is a dictionary with keys: bottom, left, top, right

    Returns x, y
    """

    # Maps Unif[0, 1] -> Unif[domain_bounds["left"], domain_bounds["right"]]
    x = (domain_bounds["right"] - domain_bounds["left"]) * torch.rand(num_interior_points) + domain_bounds["left"]
    y = (domain_bounds["top"] - domain_bounds["bottom"]) * torch.rand(num_interior_points) + domain_bounds["bottom"]

    x = x.unsqueeze(1).float()
    y = y.unsqueeze(1).float()
    x.requires_grad = True
    y.requires_grad = True
    return x, y


def gen_boundary_points_rectangle(num_boundary_points: int = 1000, domain_bounds: Dict[str, float] = {"bottom": 0, "left": 0, "top": 4, "right": 2}):
    """
    Generates boundary points on rectangular boundary
    """

    # Maps Unif[0, 1] -> Unif[domain_bounds["left"], domain_bounds["right"]]
    x = (domain_bounds["right"] - domain_bounds["left"]) * torch.rand(num_boundary_points) + domain_bounds["left"]
    y = (domain_bounds["top"] - domain_bounds["bottom"]) * torch.rand(num_boundary_points) + domain_bounds["bottom"]
    x = x.unsqueeze(1).float()
    y = y.unsqueeze(1).float()

    # Generates samples (x={left}, y), (x={right}, y), (x, y={bottom}), (x, y={top})
    y_bottom = torch.tensor(float(domain_bounds["bottom"]), dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    y_top = torch.tensor(float(domain_bounds["top"]), dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    x_left = torch.tensor(float(domain_bounds["left"]), dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)
    x_right = torch.tensor(float(domain_bounds["right"]), dtype=torch.float32, requires_grad=True).repeat(num_boundary_points).unsqueeze(1)

    bottom = torch.cat((x, y_bottom), dim=-1)
    top = torch.cat((x, y_top), dim=-1)
    left = torch.cat((x_left, y), dim=-1)
    right = torch.cat((x_right, y), dim=-1)
    return bottom, top, left, right



def gen_boundary_points_circle(radius, center_x, center_y, point_count):
    """
    Generate boundary points on a circle
    """

    t = torch.linspace(0, 2*np.pi, point_count)
    x = center_x + radius*torch.cos(t)
    y = center_y + radius*torch.sin(t)

    
    return x, y


def gen_interior_points_circle(radius, center_x, center_y, point_count):
    """
    Generate interior points on a circle

    Inverse CDF sampling for radius
    """

    radii = torch.sqrt(torch.rand(point_count)) * radius
    theta = torch.rand(point_count) * 2 * np.pi

    x = center_x + radii * torch.cos(theta)
    y = center_y + radii * torch.sin(theta)

    x = x.unsqueeze(1).float()
    y = y.unsqueeze(1).float()
    x.requires_grad = True
    y.requires_grad = True

    return x, y

radius = 1.0
center_x = 0.0
center_y = 0.0
point_count = 1000

x, y = gen_interior_points_circle(radius, center_x, center_y, point_count)
b_x, b_y = gen_boundary_points_circle(radius, center_x, center_y, point_count)

plt.figure(figsize=(8, 8))
plt.scatter(x.detach().numpy(), y.detach().numpy(), color="red")
plt.scatter(b_x.detach().numpy(), b_y.detach().numpy(), color="blue")
plt.show()




