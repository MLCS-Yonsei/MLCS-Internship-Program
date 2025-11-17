import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



# equation of motion
def f(e, u):
    r, alpha, phi = e
    v, omega = u
    return np.array([
        -v * np.cos(alpha),
        v * np.sin(alpha) / r - omega,
        -v * np.sin(alpha) / r
    ])

# controller design
lambda_r = 1.0
lambda_alpha = 1.0
lambda_phi = 1.0
k_v = 2.0
k_omega = 3.0
def k(e):
    r, alpha, phi = e
    if alpha > 1e-3:
        return np.array([
            k_v * r * np.cos(alpha),
            k_v * (lambda_alpha * alpha - lambda_phi * phi) * np.cos(alpha) * np.sin(alpha) / lambda_alpha / alpha + k_omega * alpha
        ])
    else:
        return np.array([
            k_v * r * np.cos(alpha),
            k_v * (lambda_alpha * alpha - lambda_phi * phi) * np.cos(alpha) / lambda_alpha + k_omega * alpha
        ])

# initail state
e0 = np.array([3.0, -np.pi/2, np.pi/4])

# simulation
sol = solve_ivp(
    lambda t, e: f(e, k(e)),  # closed loop system
    (0.0, 5.0),  # simulation time interval
    e0,  # initial state
    t_eval=np.linspace(0.0, 5.0, 501)
)
x = np.array([
    -sol.y[0, :] * np.cos(sol.y[2, :]),
    -sol.y[0, :] * np.sin(sol.y[2, :]),
    sol.y[1, :] + sol.y[2, :]
])

# figure plot
plt.figure("error states")
for k, label in enumerate(["r", "alpha", "phi"]):
    plt.subplot(311+k)
    plt.plot(sol.t, np.zeros_like(sol.t), "k--")
    plt.plot(sol.t, sol.y[k, :])
    plt.ylabel(label)
plt.xlabel("t")
plt.figure("robot pose")
plt.plot(x[0, :], x[1, :], "k")
plt.quiver(
    x[0, ::5], x[1, ::5], np.cos(x[2, ::5]), np.sin(x[2, ::5]),
    scale=10.0, alpha=0.3, color="g"
)
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
