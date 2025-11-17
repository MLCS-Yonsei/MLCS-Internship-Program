import numpy as np
import matplotlib.pyplot as plt



# QP: minimize x.T P x + q.T x
P = np.array(
    [
        [1.0, 0.4],
        [0.4, 2.0]
    ]
)
q = np.array([0.6, 1.2])


# Solve QP
x_opt = -np.linalg.solve(P, q)  # x = -inv(P) q


# Plot result
cost = []
x_grid = np.array(np.meshgrid(np.linspace(-3, 3, 101), np.linspace(-3, 3, 101)))
for x in x_grid.reshape((2, -1)).T:
    cost.append(0.5 * x.T @ P @ x + q @ x)
plt.contourf(*x_grid, np.reshape(cost, (101, 101)), levels=16)  # plot cost function
plt.scatter(*x_opt, c="r")  # plot optimal solution
plt.show()
