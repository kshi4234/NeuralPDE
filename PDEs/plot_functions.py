import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch import nn

def plot_losses(losses: list[float]):
    plt.plot(losses)
    plt.show()


def plot_solution(model: nn.Module, low: float, high: float):
    model.eval()
    with torch.no_grad():

        x_vals = torch.linspace(low - 0.1, high + 0.1, 1000, dtype=torch.float32)
        y_vals = torch.linspace(low - 0.1, high + 0.1, 1000, dtype=torch.float32)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")    # generates 2500 points
        
        # Flatten and pass through the model
        x_flat = X.flatten().unsqueeze(1)
        y_flat = Y.flatten().unsqueeze(1)
        z = torch.cat((x_flat, y_flat), dim=-1)
        print(z.dtype)
        u = model(z)
        U = u.reshape(X.shape).numpy()  # Reshape to grid and convert to numpy for plotting
        
        # Plotting
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert X, Y, U to numpy for plotting
        ax.plot_surface(X.numpy(), Y.numpy(), U, cmap='viridis', edgecolor='k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('U')
        
        plt.show()




