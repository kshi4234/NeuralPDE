# Code for GP-PDE

import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(module_path)

import gpytorch
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

torch.manual_seed(0)

# Boundary Conditions
def u(x, y):
    """
    2.0 if y is 0 or 1, 0.0 otherwise
    """
    return 2.0 * (y == 0) + 2.0 * (y == 1) + 0.0 * x



# x = torch.linspace(0, 1, 100).unsqueeze(1)
# y = torch.linspace(0, 1, 100).unsqueeze(1)

# x_low = torch.zeros(100).unsqueeze(1)
# x_high = torch.ones(100).unsqueeze(1)

# y_low = torch.zeros(100).unsqueeze(1)
# y_high = torch.ones(100).unsqueeze(1)

# x_free_y_low = torch.cat((x, y_low), dim=-1)
# x_free_y_high = torch.cat((x, y_high), dim=-1)

# y_free_x_low = torch.cat((x_low, y), dim=-1)
# y_free_x_high = torch.cat((x_high, y), dim=-1)

# inputs = torch.cat((x_free_y_low, x_free_y_high, y_free_x_low, y_free_x_high), dim=0)

# x = inputs[:, 0]
# y = inputs[:, 1]

# z = u(x, y)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z)
# plt.show()


# sampling points
xy = torch.rand(1000, 2)

x_bottom = torch.zeros(100).unsqueeze(1)
x_top = torch.ones(100).unsqueeze(1)
y_bottom = torch.zeros(100).unsqueeze(1)
y_top = torch.ones(100).unsqueeze(1)

# Correctly sampling interior + boundary points
bottom_x_points = torch.cat((x_bottom, xy[:100, 1].unsqueeze(1)), dim=-1)
top_x_points = torch.cat((x_top, xy[:100, 1].unsqueeze(1)), dim=-1)
bottom_y_points = torch.cat((xy[:100, 0].unsqueeze(1), y_bottom), dim=-1)
top_y_points = torch.cat((xy[:100, 0].unsqueeze(1), y_top), dim=-1)

# plt.scatter(bottom_x_points[:, 0], bottom_x_points[:, 1])
# plt.scatter(top_x_points[:, 0], top_x_points[:, 1])
# plt.scatter(bottom_y_points[:, 0], bottom_y_points[:, 1])
# plt.scatter(top_y_points[:, 0], top_y_points[:, 1])
# plt.scatter(xy[:1000, 0], xy[:1000, 1])
# plt.show()



# kernel and derivatives

kernel = gpytorch.kernels.RBFKernel()  # doesn't have a grad attr (?)


