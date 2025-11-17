import numpy as np
import matplotlib.pyplot as plt
from vehicle_model import VehicleModel
from simulator import Simulator
from mpc import MPC



if __name__=="__main__":

    vx = 20.0  # 72kph
    r = 50.0
    Ts = 0.01
    N = 10

    model = VehicleModel()
    sim = Simulator(model, vx, r, Ts)
    Q = np.diag([1.0, 0.1, 0.1, 1.0])
    R = np.diag([0.1])
    mpc = MPC(model, vx, vx / r, N, Ts, Q, R)

    hist = []
    x = sim.reset()
    for t in np.arange(0.0, 10.0, Ts):

        u = mpc.solve(x)
        x = sim.step(u)
        hist.append(np.r_[t, x, u])

    hist = np.array(hist)
    for i, label in enumerate(["ey", "dey", "epsi", "depsi"]):
        plt.subplot(511+i)
        plt.plot([hist[0, 0], hist[-1, 0]], [0.0]*2, "k--", alpha=0.5)
        plt.plot(hist[:, 0], hist[:, i+1])
        plt.ylabel(label)
    plt.subplot(515)
    plt.plot([hist[0, 0], hist[-1, 0]], [0.0]*2, "k--", alpha=0.5)
    plt.plot([hist[0, 0], hist[-1, 0]], [model.lbu]*2, "r--")
    plt.plot([hist[0, 0], hist[-1, 0]], [model.ubu]*2, "r--")
    plt.plot(hist[:, 0], hist[:, 5])
    plt.ylabel("delta")
    plt.xlabel("time")
    plt.show()
