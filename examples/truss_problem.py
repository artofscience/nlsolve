from math import pi, sin

import numpy as np
from matplotlib import pyplot as plt

from constraints import GeneralizedArcLength
from core import IncrementalSolver, IterativeSolver
from utils import Structure, Point
from controllers import Adaptive
from criteria import residual_norm


class TrussProblem:
    def __init__(self, theta0: float = pi/3):
        self.theta0 = theta0

    def force(self, a):
        return (1 / np.sqrt(1 - 2 * a * sin(self.theta0) + a ** 2) - 1) * (sin(self.theta0) - a)

    def jacobian(self, a):
        return np.array([- 1 / (a ** 2 - 2 * sin(self.theta0) * a + 1) ** (1 / 2) + (
                (a - sin(self.theta0)) * (2 * a - 2 * sin(self.theta0))) / (
                                 2 * (a ** 2 - 2 * sin(self.theta0) * a + 1) ** (3 / 2)) + 1])



if __name__ == "__main__":
    truss_problem = TrussProblem(pi/2.1)

    problem = Structure(truss_problem, ixf=[0], ff=np.array([1]))

    solver = IterativeSolver(problem, GeneralizedArcLength(), residual_norm(1e-10))

    stepper = IncrementalSolver(solver)

    p0 = Point(qf=np.zeros(1), ff=np.zeros(1))

    controller = Adaptive(0.05, max=0.5, incr=1.2, decr=0.2, min=0.001)

    solution, tries = stepper(p0, controller)

    # PLOTTING

    for a in tries:
        plt.plot([i.qf for i in a],[i.ff for i in a], 'ko--', alpha=0.1)

    plt.plot([i.qf for i in solution], [i.ff for i in solution], 'ko-')

    plt.show()
