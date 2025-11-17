import numpy as np
import matplotlib.pyplot as plt



# QP: minimize x.T P x + q.T x  s.t.  Ax = b
P = np.array(
    [
        [1.0, 0.4],
        [0.4, 2.0]
    ]
)
q = np.array([0.6, 1.2])

A = np.array(
    [
        [2.0, -1.0],
    ]
)
b = np.array([1.0])


# Solve QP
invP = np.linalg.inv(P)
u_opt = -np.linalg.solve(A @ invP @ A.T, A @ invP @ q + b)  # solution for dual problem  u = argmax_u min_x x.T P x + q.T x + u.T (Ax - b)
x_opt = -invP @ (A.T @ u_opt + q)  # solution for primal problem  x = inv(P) (A.T u + q)


# Plot result
cost = []
x_grid = np.array(np.meshgrid(np.linspace(-3, 3, 101), np.linspace(-3, 3, 101)))
for x in x_grid.reshape((2, -1)).T:
    cost.append(0.5 * x.T @ P @ x + q @ x)
plt.contourf(*x_grid, np.reshape(cost, (101, 101)), levels=16)  # plot cost function
plt.plot([-1.0, 2.0], [-3.0, 3.0], "w", alpha=0.5)  # plot equality constraint
plt.scatter(*x_opt, c="r")  # plot optimal solution
plt.show()
