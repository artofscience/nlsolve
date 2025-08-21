import numpy as np
from scipy.integrate import RK45

from utils import Problem, Point
from core import Out


class DynamicsSolver:
    def __init__(self, problem: Problem, tol: float = 1e-3):
        self.problem = problem
        self.history = []
        self.tol = tol

    def __call__(self, p0: Point, c: float = 1.0, m=None,
                 t_start: float = 0.0, t_end: float = 1e3,
                 v0: float = 0.0, tol = None):

        tmp = p0.q[self.problem.ixf]
        x0 = np.hstack((tmp, v0 * np.ones_like(tmp))) if m else tmp

        args = (self.problem, p0, c, m)
        t, y = self.solver(x0, [t_start, t_end], args, self.tol if tol is None else tol)

        self.out = Out()
        self.out.solutions =  self.get_out(y, p0, m)
        self.out.time = t
        self.history.append(self.out)
        return self.out

    def solver(self, x0, time, args, tol: float = 1e-6):
        t = []
        y = []
        solver = RK45(lambda t, y: self.dynamics(t, y, *args), time[0], x0, time[1])
        while solver.status == "running":
            t.append(solver.t)
            y.append(solver.y)
            if len(t) > 1:
                if np.all(np.abs(y[-2] - y[-1]) < tol):
                    break
            solver.step()
        return t, y

    def get_out(self, y, p0, m):
        n = len(y[0]) // 2
        out = [1.0 * p0 for i in y]
        for count, element in enumerate(out):
            element.q[self.problem.ixf] = y[count][:n] if m else y[count]
            element.f[self.problem.ixp] = self.problem.nlf.force(element.q)[self.problem.ixp]
        return out

    def load_based_offset(self, p0: Point, alpha: float = 1.0):
        point = 1.0 * p0
        if self.problem.nf:
            point.f[self.problem.ixf] += alpha / 100 * self.problem.ffc
        if self.problem.np:
            point.q[self.problem.ixp] += alpha / 100 * self.problem.qpc
        return point

    @staticmethod
    def dynamics(t, x, prob, p0, c, m):
        n = len(x) // 2
        pos = 1.0 * p0.q
        pos[prob.ixf] = x[:n] if m else x
        elastic_force = prob.nlf.force(pos)[prob.ixf]
        external_load = p0.f[prob.ixf]
        tmp = external_load - elastic_force  # diff between external and elastic load
        vel = x[n:] if m else tmp / c  # velocity
        return np.hstack((vel, (tmp - c * vel) / m)) if m else vel
