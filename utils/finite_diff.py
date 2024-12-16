import numpy as np
from matplotlib import pyplot as plt

def finite_diff(x_vals, y_vals):
    # Numeric solution:
    # The coefficient matrix A is now m*n by m*n,
    # since that is the total number of points.
    # The right-hand side vector b is m*n by 1.
    A = np.zeros((len(x_vals)*len(y_vals), len(x_vals)*len(y_vals)))
    b = np.zeros(len(x_vals)*len(y_vals))

    u_left = 1
    u_right = 1
    u_bottom = 1
    u_top = 0

    for j, y in enumerate(y_vals):
        for i, x in enumerate(x_vals):
            # for convenience, calculate all indices now
            kij = j*len(x_vals) + i
            kim1j = j*len(x_vals) + i - 1
            kip1j = j*len(x_vals) + i + 1
            kijm1 = (j-1)*len(x_vals) + i
            kijp1 = (j+1)*len(x_vals) + i
            if i == 0:
                # this is the left boundary
                A[kij, kij] = 1
                b[kij] = u_left
            elif i == len(x_vals) - 1:
                # right boundary
                A[kij, kij] = 1
                b[kij] = u_right
            elif j == 0:
                # bottom boundary
                A[kij, kij] = 1
                b[kij] = u_bottom
            elif j == len(y_vals) - 1:
                # top boundary
                A[kij, kij] = 1
                b[kij] = u_top
            else:
                # coefficients for interior points, based
                # on the recursion formula
                A[kij, kim1j] = 1
                A[kij, kip1j] = 1
                A[kij, kijm1] = 1
                A[kij, kijp1] = 1
                A[kij, kij] = -4
    u = np.linalg.solve(A, b)
    u_square = np.reshape(u, (len(y_vals), len(x_vals)))
    
    print(x_vals.shape)
    print(y_vals.shape)
    print(u_square.shape)
    
    plt.contourf(x_vals, y_vals, u_square, levels=255, cmap=plt.cm.coolwarm)
    plt.colorbar(label='Temperature')
    plt.show()
    return u_square