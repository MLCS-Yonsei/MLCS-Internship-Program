import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



class Simulator:

    def __init__(self, model, vx, R, Ts) -> None:

        self.A, self.B, self.d = model.get_lti(vx, vx / R)
        self.lbu = model.lbu
        self.ubu = model.ubu

        self.vx = vx
        self.Ts = Ts
        self.reset()


    def step(self, u):
        u = np.clip(u, self.lbu, self.ubu)
        sol = solve_ivp(
            lambda t, x: self.A @ x + self.B @ u + self.d[:, 0],
            (0, self.Ts),
            self.x
        )
        self.x = sol.y[:, -1]

        return self.x.copy()


    def reset(self):
        self.x = np.zeros(4)
        self.x[0] = np.random.uniform(-12.0, 12.0)
        self.x[2] = np.random.uniform(-np.pi/4, np.pi/4)
        return self.x
