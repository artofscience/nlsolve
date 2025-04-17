from math import pi, sin

import numpy as np
from matplotlib import pyplot as plt

from constraints import NewtonRaphson, ArcLength
from core import IncrementalSolver, IterativeSolver
from utils import Structure, Point


class TrussProblem(Structure):
    theta0 = pi / 3

    def internal_load(self, a):
        return (1 / np.sqrt(1 - 2 * a * sin(self.theta0) + a ** 2) - 1) * (sin(self.theta0) - a)

    def tangent_stiffness(self, a):
        return np.array([- 1 / (a ** 2 - 2 * sin(self.theta0) * a + 1) ** (1 / 2) + (
                (a - sin(self.theta0)) * (2 * a - 2 * sin(self.theta0))) / (
                                 2 * (a ** 2 - 2 * sin(self.theta0) * a + 1) ** (3 / 2)) + 1])


class TrussProblemLoadBased(TrussProblem):

    def ff(self):
        return np.array([-2.0], dtype=float)

    def gf(self, p):
        return super().internal_load(p.qf)

    def kff(self, p):
        return super().tangent_stiffness(p.qf)


class TrussProblemMotionBased(TrussProblem):

    def qp(self):
        return np.array([3.0], dtype=float)

    def gp(self, p):
        return super().internal_load(p.qp)

    def kpp(self, p):
        return super().tangent_stiffness(p.qp)


if __name__ == "__main__":
    solver = IncrementalSolver(IterativeSolver(TrussProblemMotionBased(), NewtonRaphson()))
    solution, tries = solver(Point(qp=np.array([0.0]), fp=np.array([0.0])))

    for a in tries:
        plt.plot([i.qf for i in a], [i.ff for i in a], 'ko', alpha=0.1)

    plt.plot([i.qp for i in solution], [i.fp for i in solution], 'bo')

    for i in solution:
        plt.axvline(x=i.qp, color='b', alpha=0.1)

    solver = IncrementalSolver(IterativeSolver(TrussProblemLoadBased(), NewtonRaphson()))
    solution, tries = solver(Point(qf=np.array([0.0]), ff=np.array([0.0])))

    for a in tries:
        plt.plot([i.qf for i in a], [i.ff for i in a], 'ko', alpha=0.1)
    plt.plot([i.qf for i in solution], [i.ff for i in solution], 'ro')

    for i in solution:
        plt.axhline(y=i.ff, color='r', alpha=0.1)

    solver = IncrementalSolver(IterativeSolver(TrussProblemLoadBased(), ArcLength()))
    solution, tries = solver(Point(qf=np.array([0.0]), ff=np.array([0.0])))

    for a in tries:
        plt.plot([i.qf for i in a], [i.ff for i in a], 'ko', alpha=0.1)
    plt.plot([i.qf for i in solution], [i.ff for i in solution], 'go')
    for i in solution:
        plt.gca().add_patch(plt.Circle((i.qf, i.ff), 0.1, color='r', fill=False, alpha=0.1))

    solver = IncrementalSolver(IterativeSolver(TrussProblemMotionBased(), ArcLength()))
    solution, tries = solver(Point(qp=np.array([0.0]), fp=np.array([0.0])))

    for a in tries:
        plt.plot([i.qf for i in a], [i.ff for i in a], 'ko', alpha=0.1)
    plt.plot([i.qp for i in solution], [i.fp for i in solution], 'ko')
    for i in solution:
        plt.gca().add_patch(plt.Circle((i.qp, i.fp), 0.1, color='r', fill=False, alpha=0.1))

    plt.gca().axis('equal')
    plt.show()
