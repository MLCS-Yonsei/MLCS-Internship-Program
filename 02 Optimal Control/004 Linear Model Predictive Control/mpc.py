import numpy as np
from scipy.signal import cont2discrete
from scipy.linalg import block_diag
from  qpsolvers import solve_qp



class MPC:

    def __init__(self, model, vx, psiref, N, Ts, Q, R):


        Ac, Bc, dc = model.get_lti(vx, psiref)
        A, B, _, _, _ = cont2discrete(
            (
                Ac,
                np.hstack([Bc, dc]),
                np.eye(model.nx),
                np.zeros((model.nx, model.nu + model.nw))
            ),
            Ts
        )
        B, d = np.hsplit(B, (model.nu,))
        Q = block_diag(*(Q,)*N)
        R = block_diag(*(R,)*N)
        S = np.zeros((model.nx * N, model.nu * N))
        T = np.zeros((model.nx * N, model.nx))
        t = np.zeros((model.nx * N, 1))
        for j in range(N):
            T[j * model.nx:(j+1) * model.nx, :] = np.linalg.matrix_power(A, j+1)
            for k in range(j+1):
                S[j*model.nx:(j+1)*model.nx, k*model.nu:(k+1)*model.nu] = np.linalg.matrix_power(A, j-k) @ B
                t[j * model.nx:(j+1) * model.nx, :] += np.linalg.matrix_power(A, k) @ d
        QS = Q @ S
        self.TQS = T.T @ QS
        self.tQS = t.T @ QS
        self.P = S.T @ QS + R
        self.q = None

        self.A = None
        self.b = None

        self.G = None
        self.h = None

        self.lb = np.empty(N)
        self.lb.fill(model.lbu)
        self.ub = np.empty(N)
        self.ub.fill(model.ubu)
        self.N = N


    def solve(self, x0):
        self.update_qp(x0)
        sol = solve_qp(self.P, self.q, self.G, self.h, self.A, self.b, lb=self.lb, ub=self.ub, solver="quadprog")
        return sol[:1]


    def update_qp(self, x0):
        self.q = x0.T @ self.TQS + self.tQS
