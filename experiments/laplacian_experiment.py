import os
import sys

"""
30 epochs w/o batching essentially solves Laplace flat soln
w/ batches also solves Laplace flat soln

Seems to learn dynamics better w/o batching?
"""

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(module_path)

from utils.data_utils import (
    generate_data, generate_data_loader
)

from utils.laplacian_utils import (
    train_no_batches, train_plot, train
)

from models.NeuralPDE import (
    MLP
)

from utils.plot_utils import (
    plot_solution, make_gif
)

import torch
import torch.nn.functional as F

if __name__ == "__main__":
    Laplacian_PINN = MLP(2, 128, 1, 3, activation=F.relu)

    

    boundary_condition = {"bottom": 3, "top": 1.5, "left": 1.5, "right": 1.5}

    optimizer = torch.optim.AdamW(Laplacian_PINN.parameters(), lr=0.01)

    # Uncomment to run w/o batching
    x_train, y_train, _, _ = generate_data(10000, 0, 1)
    laplace_losses, boundary_losses = train_no_batches(
        model=Laplacian_PINN, 
        train_x=x_train,
        train_y=y_train,
        optimizer=optimizer,
        num_epochs=30,
        boundary_condition=boundary_condition,
        alpha=1.0,
        shuffle=True
    )



    # Uncomment to run w/ batching
    # train_dl, val= generate_data_loader(1000, 0, 1)
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
    plot_solution(Laplacian_PINN, low=0, high=1, id=-1, z_lim_down=0.0, z_lim_up=3.0)
    # make_gif(folder=FOLDER, name="learning_laplacian", fps=4)
    