"""
Rectangular domain with Dirichlet boundary conditions

M_a training method: L(u) = alpha * L_boundary(u) + L_interior(u)
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
    train_no_batches_rectangle, train, rectangle_loss
)

from models.NeuralPDE import (
    VanillaMLP
)

from utils.plot_utils import (
    plot_solution, make_gif
)

import torch
import torch.nn.functional as F

if __name__ == "__main__":
    Laplacian_PINN = VanillaMLP(2, 128, 1, 3, activation=F.gelu)
    

    boundary_condition = {"bottom": 1.0, "top": 0, "left": 0, "right": 0}     # this is the boundary condition
    domain_bounds = {"bottom": 0, "left": 0, "top": 2, "right": 2}            # this is the domain bounds    

    optimizer = torch.optim.AdamW(Laplacian_PINN.parameters(), lr=0.01)

    # Uncomment to run w/o batching
    x_train, y_train = gen_interior_points_rectangle(10000, domain_bounds)

    laplace_losses, boundary_losses = train_no_batches_rectangle(
        model=Laplacian_PINN, 
        train_x=x_train,
        train_y=y_train,
        optimizer=optimizer,
        num_epochs=40,
        boundary_condition=boundary_condition,
        domain_bounds=domain_bounds,
        boundary_func=rectangle_loss,
        alpha=1.0,
        shuffle=True
    )



    # Uncomment to run w/ batching
    # train_dl, val= data_loader_square(1000, 0, 1)
    # laplace_losses, boundary_losses = train(
    #     model=Laplacian_PINN, 
    #     train_dataloader=train_dl,
    #     val_dataloader=val,
    #     optimizer=optimizer,
    #     num_epochs=50,
    #     boundary_condition=boundary_condition,
    #     alpha=1.0
    # )
    
    # print(laplace_losses[:20])
    FOLDER = "experiments/images"

    print(f"Initial Laplace Loss: {laplace_losses[0]}")
    print(f"Initial Boundary Loss: {boundary_losses[0]}")
    print(f"Laplace Loss: {laplace_losses[-1]}")
    print(f"Boundary Loss: {boundary_losses[-1]}")
    plot_solution(Laplacian_PINN, low=0, high=2, id=-1)
    # make_gif(folder=FOLDER, name="learning_laplacian", fps=4)
    