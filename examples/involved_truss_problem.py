from math import pi, sin

import numpy as np
from matplotlib import pyplot as plt

from constraints import NewtonRaphson, ArcLength, NewtonRaphsonByArcLength, GeneralizedArcLength, Structure, Point
from solver import IncrementalSolver, IterativeSolver


class InvolvedTrussProblem(Structure):
    w = 0.25
    theta0 = pi / 2.5

    def internal_load(self, a1, a2):
        f1 = (1 / np.sqrt(1 - 2 * a1 * sin(self.theta0) + a1 ** 2) - 1) * (
                sin(self.theta0) - a1) - self.w * (a2 - a1)
        return np.array([f1, self.w * (a2 - a1)])

    def tangent_stiffness(self, a1):
        df1da1 = self.w - 1 / (a1 ** 2 - 2 * sin(self.theta0) * a1 + 1) ** (1 / 2) + (
                (a1 - sin(self.theta0)) * (2 * a1 - 2 * sin(self.theta0))) / (
                         2 * (a1 ** 2 - 2 * sin(self.theta0) * a1 + 1) ** (3 / 2)) + 1
        return np.array([[df1da1, -self.w], [-self.w, self.w]], dtype=float)


class InvolvedTrussProblemLoadBased(InvolvedTrussProblem):

    def ff(self):
        return np.array([0, -0.5], dtype=float)

    def internal_load_free(self, p):
        return super().internal_load(p.uf[0], p.uf[1])

    def kff(self, p):
        return super().tangent_stiffness(p.uf[0])


class InvolvedTrussProblemMotionBased(InvolvedTrussProblem):

    def up(self):
        return np.array([4.0])

    def ff(self):
        return np.array([0.0])

    def internal_load_prescribed(self, p):
        return super().internal_load(p.uf, p.up)[1]

    def internal_load_free(self, p):
        return super().internal_load(p.uf, p.up)[0]

    def kff(self, p):
        return np.array([[super().tangent_stiffness(p.uf[0])[0, 0]]])

    def kpp(self, p):
        a = np.array([super().tangent_stiffness(p.uf[0])[1, 1]])
        return a

    def kfp(self, p):
        return np.array([super().tangent_stiffness(p.uf[0])[1, 0]])

    def kpf(self, p):
        return np.array([super().tangent_stiffness(p.uf[0])[0, 1]])


if __name__ == "__main__":

    print("Load-based NR")
    constraint1 = NewtonRaphson(InvolvedTrussProblemLoadBased())
    solution_method1 = IterativeSolver(constraint1)
    solver1 = IncrementalSolver(solution_method1)
    solution1, tries1 = solver1(Point(uf=np.zeros(2), ff=np.zeros(2)))

    for a in tries1:
        plt.plot([i.uf[1] for i in a], [-i.ff[1] for i in a], 'ko', alpha=0.1)
        plt.plot([i.uf[0] for i in a], [-i.ff[1] for i in a], 'ko', alpha=0.1)
    b = [-i.ff[1] for i in solution1]
    a = [i.uf[1] for i in solution1]
    c = [i.uf[0] for i in solution1]
    plt.plot(a, b, 'ro', alpha=0.5)
    plt.plot(c, b, 'ro', alpha=0.5)
    for i in solution1:
        plt.axhline(y=i.y, color='r')

    print("Motion-based NR")
    constraint3 = NewtonRaphson(InvolvedTrussProblemMotionBased())
    solution_method3 = IterativeSolver(constraint3)
    solver3 = IncrementalSolver(solution_method3)
    solution3, tries3 = solver3(Point(uf=np.zeros(1), up=np.zeros(1), ff=np.zeros(1), fp=np.zeros(1)))

    for a in tries3:
        plt.plot([i.up for i in a], [-i.fp for i in a], 'ko', alpha=0.1)
        plt.plot([i.uf for i in a], [-i.fp for i in a], 'ko', alpha=0.1)

    plt.plot([i.up for i in solution3], [-i.fp for i in solution3], 'bo', alpha=0.5)
    plt.plot([i.uf for i in solution3], [-i.fp for i in solution3], 'bo', alpha=0.5)

    for i in solution3:
        plt.axvline(x=i.up, color='b')

    print("Load-based ARC")
    constraint2 = ArcLength(InvolvedTrussProblemLoadBased())
    solution_method2 = IterativeSolver(constraint2)
    solver2 = IncrementalSolver(solution_method2)
    solution2, tries2 = solver2(Point(uf=np.zeros(2), ff=np.zeros(2)))

    for a in tries2:
        plt.plot([i.uf[1] for i in a], [-i.ff[1] for i in a], 'ko', alpha=0.1)
        plt.plot([i.uf[0] for i in a], [-i.ff[1] for i in a], 'ko', alpha=0.1)

    b = [-i.ff[1] for i in solution2]
    a = [i.uf[1] for i in solution2]
    c = [i.uf[0] for i in solution2]

    plt.plot(a, b, 'go', alpha=0.5)
    plt.plot(c, b, 'go', alpha=0.5)

    print("Motion-based ARC")
    constraint4 = ArcLength(InvolvedTrussProblemMotionBased())
    solution_method4 = IterativeSolver(constraint4)
    solver4 = IncrementalSolver(solution_method4)
    solution4, tries4 = solver4(Point(uf=np.zeros(1), up=np.zeros(1), ff=np.zeros(1), fp=np.zeros(1)))

    for a in tries4:
        plt.plot([i.up for i in a], [-i.fp for i in a], 'ko', alpha=0.1)
        plt.plot([i.uf for i in a], [-i.fp for i in a], 'ko', alpha=0.1)

    a = [i.up for i in solution4]
    c = [i.uf for i in solution4]

    b = [-i.fp for i in solution4]
    plt.plot(a, b, 'co', alpha=0.5)
    plt.plot(c, [-i.fp for i in solution4], 'co', alpha=0.5)

    print("Load-based ARC2")
    constraint2 = NewtonRaphsonByArcLength(InvolvedTrussProblemLoadBased())
    solution_method2 = IterativeSolver(constraint2)
    solver2 = IncrementalSolver(solution_method2)
    solution2, tries2 = solver2(Point(uf=np.zeros(2), ff=np.zeros(2)))

    for a in tries2:
        plt.plot([i.uf[1] for i in a], [-i.ff[1] for i in a], 'ko', alpha=0.1)
        plt.plot([i.uf[0] for i in a], [-i.ff[1] for i in a], 'ko', alpha=0.1)

    b = [-i.ff[1] for i in solution2]
    a = [i.uf[1] for i in solution2]
    c = [i.uf[0] for i in solution2]

    plt.plot(a, b, 'go', alpha=0.5)
    plt.plot(c, b, 'go', alpha=0.5)

    print("Motion-based ARC2")
    constraint4 = NewtonRaphsonByArcLength(InvolvedTrussProblemMotionBased())
    solution_method4 = IterativeSolver(constraint4)
    solver4 = IncrementalSolver(solution_method4)
    solution4, tries4 = solver4(Point(uf=np.zeros(1), up=np.zeros(1), ff=np.zeros(1), fp=np.zeros(1)))

    for a in tries4:
        plt.plot([i.up for i in a], [-i.fp for i in a], 'ko', alpha=0.1)
        plt.plot([i.uf for i in a], [-i.fp for i in a], 'ko', alpha=0.1)

    a = [i.up for i in solution4]
    c = [i.uf for i in solution4]

    b = [-i.fp for i in solution4]
    plt.plot(a, b, 'ko', alpha=0.5)
    plt.plot(c, [-i.fp for i in solution4], 'ko', alpha=0.5)

    print("Load-based GENERAL a0")
    constraint2 = GeneralizedArcLength(InvolvedTrussProblemLoadBased(), alpha=0.0)
    solution_method2 = IterativeSolver(constraint2)
    solver2 = IncrementalSolver(solution_method2)
    solution2, tries2 = solver2(Point(uf=np.zeros(2), ff=np.zeros(2)))

    for a in tries2:
        plt.plot([i.uf[1] for i in a], [-i.ff[1] for i in a], 'ko', alpha=0.1)
        plt.plot([i.uf[0] for i in a], [-i.ff[1] for i in a], 'ko', alpha=0.1)

    b = [-i.ff[1] for i in solution2]
    a = [i.uf[1] for i in solution2]
    c = [i.uf[0] for i in solution2]

    plt.plot(a, b, 'go', alpha=0.5)
    plt.plot(c, b, 'go', alpha=0.5)

    print("Motion-based GENERAL a0")
    constraint4 = GeneralizedArcLength(InvolvedTrussProblemMotionBased(), alpha=0.0)
    solution_method4 = IterativeSolver(constraint4)
    solver4 = IncrementalSolver(solution_method4)
    solution4, tries4 = solver4(Point(uf=np.zeros(1), up=np.zeros(1), ff=np.zeros(1), fp=np.zeros(1)))

    for a in tries4:
        plt.plot([i.up for i in a], [-i.fp for i in a], 'ko', alpha=0.1)
        plt.plot([i.uf for i in a], [-i.fp for i in a], 'ko', alpha=0.1)

    a = [i.up for i in solution4]
    c = [i.uf for i in solution4]

    b = [-i.fp for i in solution4]
    plt.plot(a, b, 'ko', alpha=0.5)
    plt.plot(c, [-i.fp for i in solution4], 'ko', alpha=0.5)

    print("Load-based GENERAL a1")
    constraint2 = GeneralizedArcLength(InvolvedTrussProblemLoadBased(), alpha=1.0)
    solution_method2 = IterativeSolver(constraint2)
    solver2 = IncrementalSolver(solution_method2)
    solution2, tries2 = solver2(Point(uf=np.zeros(2), ff=np.zeros(2)))

    for a in tries2:
        plt.plot([i.uf[1] for i in a], [-i.ff[1] for i in a], 'ko', alpha=0.1)
        plt.plot([i.uf[0] for i in a], [-i.ff[1] for i in a], 'ko', alpha=0.1)

    b = [-i.ff[1] for i in solution2]
    a = [i.uf[1] for i in solution2]
    c = [i.uf[0] for i in solution2]

    plt.plot(a, b, 'go', alpha=0.5)
    plt.plot(c, b, 'go', alpha=0.5)

    print("Motion-based GENERAL a1")
    constraint4 = GeneralizedArcLength(InvolvedTrussProblemMotionBased(), alpha=1.0)
    solution_method4 = IterativeSolver(constraint4)
    solver4 = IncrementalSolver(solution_method4)
    solution4, tries4 = solver4(Point(uf=np.zeros(1), up=np.zeros(1), ff=np.zeros(1), fp=np.zeros(1)))

    for a in tries4:
        plt.plot([i.up for i in a], [-i.fp for i in a], 'ko', alpha=0.1)
        plt.plot([i.uf for i in a], [-i.fp for i in a], 'ko', alpha=0.1)

    a = [i.up for i in solution4]
    c = [i.uf for i in solution4]

    b = [-i.fp for i in solution4]
    plt.plot(a, b, 'ko', alpha=0.5)
    plt.plot(c, [-i.fp for i in solution4], 'ko', alpha=0.5)

    plt.gca().axis('equal')
    plt.show()
