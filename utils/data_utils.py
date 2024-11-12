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

    x = torch.tensor(np.linspace(low, high, num_points).reshape(-1, 1).astype(np.float32), requires_grad=True)
    y = torch.tensor(np.linspace(low, high, num_points).reshape(-1, 1).astype(np.float32), requires_grad=True)

    x_val = torch.tensor(np.linspace(low, high, num_points // 10).reshape(-1, 1).astype(np.float32), requires_grad=True)
    y_val = torch.tensor(np.linspace(low, high, num_points // 10).reshape(-1, 1).astype(np.float32), requires_grad=True)

    return x, y, x_val, y_val
