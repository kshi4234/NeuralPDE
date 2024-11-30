import os
import sys
import gpytorch
import matplotlib.pyplot as plt

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(module_path)

from utils.data_utils import (
    gen_interior_points_rectangle
)

from utils.pde_utils import train_no_batches, compute_boundary_loss_rectangle

from utils.laplacian_utils import (
    compute_laplacian
)

from models.NeuralPDE import (
    VanillaMLP
)

from utils.plot_utils import (
    plot_solution
)

import torch
import torch.nn.functional as F


# Each "experiment" should define a unique loss function
def laplacian_dirichlet_rectangle_loss(model, u, x, y, boundary_condition, domain_bounds):
    laplacian = compute_laplacian(u, x, y)
    laplacian_loss = F.mse_loss(laplacian, torch.zeros_like(laplacian))

    boundary_loss = compute_boundary_loss_rectangle(model, domain_bounds, boundary_condition, 25)

    return laplacian_loss, boundary_loss

if __name__ == "__main__":

    torch.manual_seed(42)   # Reproducibility

    Laplacian_PINN = VanillaMLP(2, 128, 1, 3, activation=F.gelu)

    high = 3.0

    
    boundary_condition = {"bottom": 1.0, "top": 2.0, "left": 0.0, "right": 0}     # this is the boundary condition
    domain_bounds = {"bottom": 0, "left": 0, "top": high, "right": high}            # this is the domain bounds    

    optimizer = torch.optim.AdamW(Laplacian_PINN.parameters(), lr=0.01)

    x_train, y_train = gen_interior_points_rectangle(10000, domain_bounds)

    laplace_losses, boundary_losses = train_no_batches(
        model=Laplacian_PINN, 
        train_x=x_train,
        train_y=y_train,
        optimizer=optimizer,
        loss_fn=laplacian_dirichlet_rectangle_loss,
        boundary_condition=boundary_condition,
        domain_bounds=domain_bounds,
        num_epochs=50,
        shuffle=True
    )

    plot_solution(Laplacian_PINN, low=0, high=high, id=-1)
    # make_gif(folder=FOLDER, name="learning_laplacian", fps=4)