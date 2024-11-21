import matplotlib.pyplot as plt

import torch
from torch import nn
import os
from typing import Union, Optional
import logging


import imageio

FOLDER = "experiments/images"
os.makedirs(FOLDER, exist_ok=True)


def extract_num(
    filename: str
) -> Union[int, float]:
    """
    Extracts number from filename

    Helper function for make_gif
    """
    base = os.path.basename(filename)
    number = ''.join(filter(str.isdigit, base))
    return int(number) if number.isdigit() else float('inf')


def make_gif(
    folder: str,
    name: str,
    fps: int
) -> None:
    """
    Construct gif 'name' from the images in 'folder'

    Args:
        folder: path to gif folder
        name: name of gif to make
        fps: frames per second
    """
    logging.info(f"Making {name}.gif from images in {folder}")

    with imageio.get_writer(f'{name}.gif', mode = 'I', fps = fps, loop=0) as writer:
        for filename in sorted(os.listdir(folder), key=extract_num):
            if filename.endswith('png'):
                image = imageio.imread(folder+"/"+filename)
                writer.append_data(image)




def plot_losses(losses: list[float]):
    plt.plot(losses)
    plt.show()


def plot_solution(model: nn.Module, low: float, high: float, id: Optional[int] = 0):
    """
    Evaluates model over grid of points to visualize surface
    
    pass id = -1 to view instead of saving
    """
    model.eval()
    with torch.no_grad():

        x_vals = torch.linspace(low, high, 1000, dtype=torch.float32)
        y_vals = torch.linspace(low, high, 1000, dtype=torch.float32)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")    # generates 2500 points
        
        # Flatten and pass through the model
        x_flat = X.flatten().unsqueeze(1)
        y_flat = Y.flatten().unsqueeze(1)
        z = torch.cat((x_flat, y_flat), dim=-1)
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
        # ax.set_zlim(2, 3)
        
        if id == -1:
            plt.show()
            print(U)
        
        else:
            plt.savefig(f"{FOLDER}/{str(id)}.png")
            plt.close()




