from math import pi, sin

import numpy as np
from matplotlib import pyplot as plt

from constraints import NewtonRaphson, ArcLength
from point import Point
from solver import IncrementalSolver, IterativeSolver
from structure import Structure


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

    def internal_load_free(self, p):
        return super().internal_load(p.uf)

    def kff(self, p):
        return super().tangent_stiffness(p.uf)


class TrussProblemMotionBased(TrussProblem):

    def up(self):
        return np.array([3.0], dtype=float)

    def internal_load_prescribed(self, p):
        return super().internal_load(p.up)

    def kpp(self, p):
        return super().tangent_stiffness(p.up)


if __name__ == "__main__":
    constraint1 = NewtonRaphson(TrussProblemMotionBased())
    solution_method1 = IterativeSolver(constraint1)
    solver1 = IncrementalSolver(solution_method1)
    solution1, tries1 = solver1(Point(up=np.array([0.0]), fp=np.array([0.0])))

    for a in tries1:
        plt.plot([i.uf for i in a], [i.ff for i in a], 'ko', alpha=0.1)

    plt.plot([i.up for i in solution1], [i.fp for i in solution1], 'bo')

    for i in solution1:
        plt.axvline(x=i.up, color='b', alpha=0.1)

    constraint2 = NewtonRaphson(TrussProblemLoadBased())
    solution_method2 = IterativeSolver(constraint2)
    solver2 = IncrementalSolver(solution_method2)
    solution2, tries2 = solver2(Point(uf=np.array([0.0]), ff=np.array([0.0])))

    for a in tries2:
        plt.plot([i.uf for i in a], [i.ff for i in a], 'ko', alpha=0.1)
    plt.plot([i.uf for i in solution2], [i.ff for i in solution2], 'ro')

    for i in solution2:
        plt.axhline(y=i.ff, color='r', alpha=0.1)

    constraint3 = ArcLength(TrussProblemLoadBased())
    solution_method3 = IterativeSolver(constraint3)
    solver3 = IncrementalSolver(solution_method3)
    solution3, tries3 = solver3(Point(uf=np.array([0.0]), ff=np.array([0.0])))

    for a in tries3:
        plt.plot([i.uf for i in a], [i.ff for i in a], 'ko', alpha=0.1)
    plt.plot([i.uf for i in solution3], [i.ff for i in solution3], 'go')
    for i in solution3:
        plt.gca().add_patch(plt.Circle((i.uf, i.ff), 0.1, color='r', fill=False, alpha=0.1))

    constraint4 = ArcLength(TrussProblemMotionBased())
    solution_method4 = IterativeSolver(constraint4)
    solver4 = IncrementalSolver(solution_method4)
    solution4, tries4 = solver4(Point(up=np.array([0.0]), fp=np.array([0.0])))

    for a in tries4:
        plt.plot([i.uf for i in a], [i.ff for i in a], 'ko', alpha=0.1)
    plt.plot([i.up for i in solution4], [i.fp for i in solution4], 'ko')
    for i in solution4:
        plt.gca().add_patch(plt.Circle((i.up, i.fp), 0.1, color='r', fill=False, alpha=0.1))

    plt.gca().axis('equal')
    plt.show()
