from models.NeuralPDE import DomainDataset
from torch.utils.data import DataLoader
import torch
import numpy as np



def generate_data_loader(
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


def generate_data(
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



def gen_boundary_points(num_boundary_points: int = 1000, low: int = 0, high: int = 1):
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