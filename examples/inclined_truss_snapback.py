from math import pi, sin

import numpy as np
from matplotlib import pyplot as plt

from constraints import GeneralizedArcLength
from core import IncrementalSolver, IterativeSolver
from utils import Structure, Point
from controllers import Adaptive
from criteria import residual_norm

"""
Analysis of two-DOF inclined truss with severe snapback behaviour.
"""

class InclinedTrussSnapback(Structure):
    def __init__(self, w: float = 0.1, theta0: float = pi/2.5):
        self.w = w
        self.theta0 = theta0

    def force(self, a):
        a1, a2 = a[0], a[1]
        f1 = (1 / np.sqrt(1 - 2 * a1 * sin(self.theta0) + a1 ** 2) - 1) * (
                sin(self.theta0) - a1) - self.w * (a2 - a1)
        return np.array([f1, self.w * (a2 - a1)])

    def jacobian(self, a):
        a1 = a[0]
        df1da1 = self.w - 1 / (a1 ** 2 - 2 * sin(self.theta0) * a1 + 1) ** (1 / 2) + (
                (a1 - sin(self.theta0)) * (2 * a1 - 2 * sin(self.theta0))) / (
                         2 * (a1 ** 2 - 2 * sin(self.theta0) * a1 + 1) ** (3 / 2)) + 1
        return np.array([[df1da1, -self.w], [-self.w, self.w]], dtype=float)


if __name__ == "__main__":
    truss = InclinedTrussSnapback()

    problem = Structure(truss, ixf=[0, 1], ff=np.array([0, 0.5]))

    solver = IterativeSolver(problem, GeneralizedArcLength(), residual_norm(1e-6))

    stepper = IncrementalSolver(solver)

    p0 = Point(qf=np.array([0, 0]), ff=np.array([0, 0]))

    controller = Adaptive(0.1, max=0.5, incr=1.2, decr=0.1, min=0.0001)

    solution = stepper(p0, controller)[0]

    # PLOTTING

    # plot both DOF 0 and 1 wrt the loading magnitude at DOF 1
    plt.plot([i.qf[0] for i in solution], [i.ff[1] for i in solution], 'ko-')
    plt.plot([i.qf[1] for i in solution], [i.ff[1] for i in solution], 'bo-')

    plt.show()
