"""
Rectangular domain with Dirichlet boundary conditions

M_a training method: L(u) = alpha * L_boundary(u) + L_interior(u)
"""

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

    # Train on random points
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

    # Test on a grid - 100 points for GP uncertainty visualization
    x_test, y_test = torch.linspace(0, high, 10, dtype=torch.float32), torch.linspace(0, high, 50, dtype=torch.float32)
    
    X, Y = torch.meshgrid(x_test, y_test, indexing="ij")

    x_test_flat = X.flatten().unsqueeze(1)
    y_test_flat = Y.flatten().unsqueeze(1)

    # Generating targets as output of PINN
    xy_test = torch.cat((x_test_flat, y_test_flat), dim=-1)

    # This needs to be flattened and detached 
    u_test = Laplacian_PINN(xy_test).squeeze(1)
    # Add noise
    u_test += torch.randn_like(u_test) * 0.20

    u_plot = u_test.reshape(X.shape).detach().numpy()

    # 3D Plotting
    with torch.no_grad():
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(X.numpy(), Y.numpy(), u_plot,cmap='viridis', edgecolor='k')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('U')
        plt.title("3D Scatter Plot of PINN Output")
        plt.show()

    # Flatten and detach
    u_test = u_test.flatten().detach()

    # GP Model
    class GPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean = self.mean_module(x)
            cov_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, cov_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = GPModel(xy_test, u_test, likelihood)

    # Training
    gp_model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)
    marginal_ll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    for i in range(100):
        optimizer.zero_grad()
        output = gp_model(xy_test)
        loss = -marginal_ll(output, u_test)
        loss.backward()
        optimizer.step()    

        if (i + 1) % 10 == 0:
            print(f'Iter {i+1}/{100} - Loss: {loss.item()}, -Lengthscale: {gp_model.covar_module.base_kernel.lengthscale.item()}, noise: {likelihood.noise.item()}')
    
    # Predictions
    gp_model.eval()
    likelihood.eval()

    # Generate a grid 
    # TODO: plot_trisurf
    GP_x, GP_y = torch.linspace(0, high, 5, dtype=torch.float32), torch.linspace(0, high, 5, dtype=torch.float32)
    # GP_x, GP_y = GP_x.squeeze(1), GP_y.squeeze(1)

    GPX, GPY = torch.meshgrid(GP_x, GP_y, indexing="ij")

    GPX_FLAT = GPX.flatten().unsqueeze(1)
    GPY_FLAT = GPY.flatten().unsqueeze(1)

    GP_XY = torch.cat((GPX_FLAT, GPY_FLAT), dim=-1)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_preds = gp_model(GP_XY)
        f_samples = f_preds.sample(sample_shape=torch.Size([3]))
        lower, upper = f_preds.confidence_region()
    

    with torch.no_grad():
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(projection="3d")   

        print(GPX.shape)
        print(GPX)
        print(GPY.shape)
        print(GPY)
        print(f_samples[0].reshape(GPX.shape).shape)
        # Sample functions
        colors = ['red', 'green', 'blue']
        for i in range(f_samples.shape[0]):
            ax.plot_surface(
                GPX.numpy(),
                GPY.numpy(),
                f_samples[i].reshape(GPX.shape).numpy(),
                alpha=0.4,
                edgecolor='none',
                color=colors[i]
            )

        # Confidence region
        lower, upper = f_preds.confidence_region()
        # lower_surface = lower.reshape(GPX.shape).numpy() - 3.0
        # upper_surface = upper.reshape(GPX.shape).numpy() + 3.0

        # ax.plot_surface(GPX.numpy(), GPY.numpy(), lower_surface, alpha=0.5, color="blue", label="Lower Bound")
        # ax.plot_surface(GPX.numpy(), GPY.numpy(), upper_surface, alpha=0.5, color="red", label="Upper Bound")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.title("Laplacian Dirichlet GP Solution")
        plt.legend(["Observed Data", "Sample Functions"])
        plt.show()

    





    


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
    # FOLDER = "experiments/images"

    # print(f"Initial Laplace Loss: {laplace_losses[0]}")
    # print(f"Initial Boundary Loss: {boundary_losses[0]}")
    # print(f"Laplace Loss: {laplace_losses[-1]}")
    # print(f"Boundary Loss: {boundary_losses[-1]}")
    # plot_solution(Laplacian_PINN, low=0, high=high, id=-1)
    # make_gif(folder=FOLDER, name="learning_laplacian", fps=4)
    