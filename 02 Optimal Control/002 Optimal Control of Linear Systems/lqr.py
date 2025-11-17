import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt



# system matrices
A = np.array([
    [ 0.0,  0.0,  1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0],
    [-3.0,  2.0, -2.0,  1.0],
    [ 1.0, -1.0,  0.5, -1.0]
])
B = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 0.5]
])
n = A.shape[0]

# continuous-time lqr
Q = np.diag([1e2, 1e2, 1e0, 1e0])
R = 1e-2*np.eye(2)
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# initail state and reference
x0 = np.array([0.0, 0.0, 0.0, 0.0])
xref = np.array([0.5, 1.0, 0.0, 0.0])

# simulation
sol = solve_ivp(
    lambda t, x: A @ x + B @ K @ (xref - x),  # closed loop system
    (0.0, 5.0),  # simulation time interval
    x0,  # initial state
    t_eval=np.linspace(0.0, 5.0, 501)
)

# discrete-time system
Ts = 0.05  # sampling time
Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(n), np.zeros_like(B)), Ts)

# discrete-time lqr
Pd = solve_discrete_are(Ad, Bd, Q, R)
Kd = np.linalg.inv(R + Bd.T @ Pd @ Bd) @ Bd.T @ Pd @ Ad

# simulation
t = np.arange(0.0, 5.0, Ts)  # simulation time sequence
y = [x0.copy()]  # list of system state
for _ in t[1:]:
    y.append(Ad @ y[-1] + Bd @ Kd @ (xref - y[-1]))
y = np.array(y).T


# figure plot
plt.figure("continuous-time system")
for k, label in enumerate(["p1", "p2", "v1", "v2"]):
    plt.subplot(411+k)
    plt.plot(sol.t, xref[k] * np.ones_like(sol.t), "k--")
    plt.plot(sol.t, sol.y[k, :])
    plt.legend([label+"ref", label])
plt.xlabel("t")
plt.figure("discrete-time system")
for k, label in enumerate(["p1", "p2", "v1", "v2"]):
    plt.subplot(411+k)
    plt.plot(t, xref[k] * np.ones_like(t), "k--")
    plt.plot(t, y[k, :])
    plt.legend([label+"ref", label])
plt.xlabel("t")
plt.show()
