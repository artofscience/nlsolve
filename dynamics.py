from utils import Problem, Point
from scipy.integrate import solve_ivp
import numpy as np

class DynamicsSolver:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.history = []

    def __call__(self, p0: Point, c: float = 1.0, m = None, t_start: float = 0.0, t_end: float = 100.0, alpha: float = 0.0, v0: float = 0.0):
        p = 1.0 * p0

        p.f[self.problem.ixf] += alpha / 100 * self.problem.ffc
        self.f0 = p.f[self.problem.ixf]

        if self.problem.ixp:
            p.q[self.problem.ixp] += alpha / 100 * self.problem.qpc
        x0 = p.q[self.problem.ixf]
        self.x0 = np.hstack((x0, v0 * np.ones_like(x0))) if m else x0

        sol = solve_ivp(self.dynamics, [0, t_end], self.x0, args=(self.problem, p.q, self.f0, c, m))
        self.history.append(sol)
        return sol

    @staticmethod
    def dynamics(t, x, prob, q0, f, c, m):
        n = len(x)//2
        pos = 1.0 * q0
        pos[prob.ixf] = x[:n] if m else x
        elastic_force = prob.nlf.force(pos)[prob.ixf]
        tmp = f - elastic_force # diff between external and elastic load
        vel = x[n:] if m else tmp / c # velocity
        return np.hstack((vel, (tmp - c * vel) / m)) if m else vel