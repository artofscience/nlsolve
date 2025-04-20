from math import pi, sin

import numpy as np
from matplotlib import pyplot as plt

from constraints import NewtonRaphson, ArcLength
from core import IncrementalSolver, IterativeSolver
from utils import Structure, Point


class TrussProblem:
    theta0 = pi / 3

    def force(self, a):
        return (1 / np.sqrt(1 - 2 * a * sin(self.theta0) + a ** 2) - 1) * (sin(self.theta0) - a)

    def jacobian(self, a):
        return np.array([- 1 / (a ** 2 - 2 * sin(self.theta0) * a + 1) ** (1 / 2) + (
                (a - sin(self.theta0)) * (2 * a - 2 * sin(self.theta0))) / (
                                 2 * (a ** 2 - 2 * sin(self.theta0) * a + 1) ** (3 / 2)) + 1])



if __name__ == "__main__":
    prob = Structure(TrussProblem(), ixf=[0, 1], ff=np.array([0, 1]))
    solver = IncrementalSolver(IterativeSolver(prob, NewtonRaphson()))
    solution, tries = solver(Point(qp=np.array([0.0]), fp=np.array([0.0])))

    for a in tries:
        plt.plot([i.qf for i in a], [i.ff for i in a], 'ko', alpha=0.1)

    plt.plot([i.qp for i in solution], [i.fp for i in solution], 'bo')

    for i in solution:
        plt.axvline(x=i.qp, color='b', alpha=0.1)

    plt.gca().axis('equal')
    plt.show()
