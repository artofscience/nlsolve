import numpy as np
from math import pi, sin
from solver import IncrementalSolver, IterativeSolver, Structure
from matplotlib import pyplot as plt


class TrussProblemLoadBased(Structure):
    theta0 = pi/3

    def external_load(self):
        return np.array([-2.0], dtype=float)

    def internal_load_free(self, p):
        return (1 / np.sqrt(1 - 2 * p.u * sin(self.theta0) + p.u ** 2) - 1) * (sin(self.theta0) - p.u)

    def tangent_stiffness_free_free(self, p):
        return np.array([- 1 / (p.u ** 2 - 2 * sin(self.theta0) * p.u + 1) ** (1 / 2) + (
                (p.u - sin(self.theta0)) * (2 * p.u - 2 * sin(self.theta0))) / (
                         2 * (p.u ** 2 - 2 * sin(self.theta0) * p.u + 1) ** (3 / 2)) + 1])


class TrussProblemMotionBased(Structure):
    theta0 = pi/3

    def prescribed_motion(self):
        return np.array([3.0], dtype=float)

    def internal_load_prescribed(self, p):
        return (1 / np.sqrt(1 - 2 * p.v * sin(self.theta0) + p.v ** 2) - 1) * (sin(self.theta0) - p.v)

    def tangent_stiffness_prescribed_prescribed(self, p):
        state = p.v
        return np.array([- 1 / (state ** 2 - 2 * sin(self.theta0) * state + 1) ** (1 / 2) + (
                (state - sin(self.theta0)) * (2 * state - 2 * sin(self.theta0))) / (
                         2 * (state ** 2 - 2 * sin(self.theta0) * state + 1) ** (3 / 2)) + 1])


if __name__ == "__main__":
    solution_method1 = IterativeSolver(TrussProblemMotionBased(), al=False)
    solver1 = IncrementalSolver(solution_method1)
    solution1, tries1 = solver1()

    for a in tries1:
        plt.plot([i.u for i in a], [i.f for i in a], 'ko', alpha=0.1)

    plt.plot([i.v for i in solution1], [i.p for i in solution1], 'bo')

    for i in solution1:
        plt.axvline(x=i.v, color='b', alpha=0.1)

    solution_method2 = IterativeSolver(TrussProblemLoadBased(), al=False)
    solver2 = IncrementalSolver(solution_method2)
    solution2, tries2 = solver2()

    for a in tries2:
        plt.plot([i.u for i in a], [i.f for i in a], 'ko', alpha=0.1)
    plt.plot([i.u for i in solution2], [i.f for i in solution2], 'ro')

    for i in solution2:
        plt.axhline(y=i.f, color='r', alpha=0.1)

    solution_method3 = IterativeSolver(TrussProblemLoadBased())
    solver3 = IncrementalSolver(solution_method3, alpha=0.3)
    solution3, tries3 = solver3()


    for a in tries3:
        plt.plot([i.u for i in a], [i.f for i in a], 'ko', alpha=0.1)
    plt.plot([i.u for i in solution3], [i.f for i in solution3], 'go')
    for i in solution3:
        plt.gca().add_patch(plt.Circle((i.u, i.f), 0.3, color='r', fill=False, alpha=0.1))

    solution_method4 = IterativeSolver(TrussProblemMotionBased())
    solver4 = IncrementalSolver(solution_method4, alpha=0.3)
    solution4, tries4 = solver4()

    for a in tries4:
        plt.plot([i.u for i in a], [i.f for i in a], 'ko', alpha=0.1)
    plt.plot([i.v for i in solution4], [i.p for i in solution4], 'ko')
    for i in solution4:
        plt.gca().add_patch(plt.Circle((i.v, i.p), 0.3, color='r', fill=False, alpha=0.1))

    plt.gca().axis('equal')
    plt.show()