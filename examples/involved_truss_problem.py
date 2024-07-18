import numpy as np
from math import pi, sin
from solver import IncrementalSolver, IterativeSolver, Structure
from matplotlib import pyplot as plt


class InvolvedTrussProblemLoadBased(Structure):
    w = 1.0
    theta0 = pi/3

    def external_load(self):
        return np.array([0, -1.0], dtype=float)

    def internal_load_free(self, p):
        f1 = (1 / np.sqrt(1 - 2 * p.u[0] * sin(self.theta0) + p.u[0] ** 2) - 1) * (sin(self.theta0) - p.u[0]) - self.w * (p.u[1] - p.u[0])
        return np.array([f1, self.w * (p.u[1] - p.u[0])])

    def tangent_stiffness_free_free(self, p):
        a1 = p.u[0]
        df1da1 = self.w - 1 / (a1 ** 2 - 2 * sin(self.theta0) * a1 + 1) ** (1 / 2) + (
                (a1 - sin(self.theta0)) * (2 * a1 - 2 * sin(self.theta0))) / (
                         2 * (a1 ** 2 - 2 * sin(self.theta0) * a1 + 1) ** (3 / 2)) + 1
        return np.array([[df1da1, -self.w], [-self.w, self.w]], dtype=float)


class InvolvedTrussProblemMotionBased(InvolvedTrussProblemLoadBased):

    def prescribed_motion(self):
        return np.array([3.0])

    def external_load(self):
        return np.array([0.0])

    def internal_load_prescribed(self, p):
        return self.w * (p.v - p.u)

    def internal_load_free(self, p):
        return (1 / np.sqrt(1 - 2 * p.u * sin(self.theta0) + p.u ** 2) - 1) * (sin(self.theta0) - p.u) - self.w * (p.v - p.u)

    def tangent_stiffness_free_free(self, p):
        return np.array([self.w - 1 / (p.u ** 2 - 2 * sin(self.theta0) * p.u + 1) ** (1 / 2) + (
                (p.u - sin(self.theta0)) * (2 * p.u - 2 * sin(self.theta0))) / (
                         2 * (p.u ** 2 - 2 * sin(self.theta0) * p.u + 1) ** (3 / 2)) + 1])

    def tangent_stiffness_prescribed_prescribed(self, p):
        return np.array([self.w])

    def tangent_stiffness_free_prescribed(self, p):
        return np.array([-self.w])

    def tangent_stiffness_prescribed_free(self, p):
        return np.array([-self.w])


if __name__ == "__main__":
    solution_method1 = IterativeSolver(InvolvedTrussProblemLoadBased(), al=False)
    solver1 = IncrementalSolver(solution_method1)
    solution1 = solver1()

    b = [-i.f[1] for i in solution1]
    a = [i.u[1] for i in solution1]
    plt.plot(a, b, 'ro')
    for i in solution1:
        plt.axhline(y=i.y, color='r')

    solution_method2 = IterativeSolver(InvolvedTrussProblemLoadBased())
    solver2 = IncrementalSolver(solution_method2, alpha=0.2)
    solution2 = solver2()

    b = [-i.f[1] for i in solution2]
    a = [i.u[1] for i in solution2]
    plt.plot(a, b, 'go')

    solution_method3 = IterativeSolver(InvolvedTrussProblemMotionBased(), al=False)
    solver3 = IncrementalSolver(solution_method3)
    solution3 = solver3()

    a = [i.v for i in solution3]
    b = [-i.p for i in solution3]
    plt.plot(a, b, 'bo')
    for i in solution3:
        plt.axvline(x=i.v, color='b')

    solution_method4 = IterativeSolver(InvolvedTrussProblemMotionBased())
    solver4 = IncrementalSolver(solution_method4, alpha=0.2)
    solution4 = solver4()

    a = [i.v for i in solution4]
    b = [-i.p for i in solution4]
    plt.plot(a, b, 'bo')

    plt.show()
