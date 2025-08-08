from utils import Problem, Point
from scipy.integrate import solve_ivp
import numpy as np

class DynamicsSolver:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.history = []

    def __call__(self, p0: Point, c: float = 1.0, m = None, t_start: float = 0.0, t_end: float = 100.0, alpha: float = 0.0, v0: float = 0.0):
        x0 = p0.q[self.problem.ixf]
        self.x0 = np.hstack((x0, v0 * np.ones_like(x0))) if m else x0
        self.f0 = (1 + alpha/100) * p0.f[self.problem.ixf]
        sol = solve_ivp(self.dynamics, [0, t_end], self.x0, args=(self.problem, p0.q, self.f0, c, m))
        self.history.append(sol)
        return sol

    @staticmethod
    def dynamics(t, x, prob, q0, f, c, m):
        pos = 1.0 * q0
        pos[prob.ixf] = x[:len(x)//2] if m else x
        elastic_force = prob.nlf.force(pos)
        tmp = f - elastic_force # diff between external and elastic force
        vel = x[len(x) // 2:] if m else None
        damping_force = c * vel if m else None
        return np.hstack((vel, (tmp - damping_force) / m)) if m else tmp / c