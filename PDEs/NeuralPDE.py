import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



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