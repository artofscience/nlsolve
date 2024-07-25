import numpy as np
from math import pi, sin
from solver import NewtonRaphson, ArcLength
from solver import IncrementalSolver, IterativeSolver, Structure, Point
from matplotlib import pyplot as plt


class InvolvedTrussProblemLoadBased(Structure):
    w = 0.5
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
        return np.array([4.0])

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

    print("Load-based NR")
    constraint1 = NewtonRaphson(InvolvedTrussProblemLoadBased())
    solution_method1 = IterativeSolver(constraint1)
    solver1 = IncrementalSolver(solution_method1)
    solution1, tries1 = solver1(Point(u=np.zeros(2), f=np.zeros(2)))

    for a in tries1:
        plt.plot([i.u[1] for i in a], [-i.f[1] for i in a], 'ko', alpha=0.1)
        plt.plot([i.u[0] for i in a], [-i.f[1] for i in a], 'ko', alpha=0.1)
    b = [-i.f[1] for i in solution1]
    a = [i.u[1] for i in solution1]
    c = [i.u[0] for i in solution1]
    plt.plot(a, b, 'ro')
    plt.plot(c, b, 'ro')
    for i in solution1:
        plt.axhline(y=i.y, color='r')

    print("Load-based ARC")
    constraint2 = ArcLength(InvolvedTrussProblemLoadBased())
    solution_method2 = IterativeSolver(constraint2)
    solver2 = IncrementalSolver(solution_method2)
    solution2, tries2 = solver2(Point(u=np.zeros(2), f=np.zeros(2)))

    for a in tries2:
        plt.plot([i.u[1] for i in a], [-i.f[1] for i in a], 'ko', alpha=0.1)
        plt.plot([i.u[0] for i in a], [-i.f[1] for i in a], 'ko', alpha=0.1)


    b = [-i.f[1] for i in solution2]
    a = [i.u[1] for i in solution2]
    c = [i.u[0] for i in solution2]

    plt.plot(a, b, 'go')
    plt.plot(c, b, 'go')

    print("Motion-based NR")
    constraint3 = NewtonRaphson(InvolvedTrussProblemMotionBased())
    solution_method3 = IterativeSolver(constraint3)
    solver3 = IncrementalSolver(solution_method3)
    solution3, tries3 = solver3(Point(u=np.zeros(1), v=np.zeros(1), f=np.zeros(1), p=np.zeros(1)))

    for a in tries3:
        plt.plot([i.v for i in a], [-i.p for i in a], 'ko', alpha=0.1)
        plt.plot([i.u for i in a], [-i.p for i in a], 'ko', alpha=0.1)

    plt.plot([i.v for i in solution3], [-i.p for i in solution3], 'bo')
    plt.plot([i.u for i in solution3], [-i.p for i in solution3], 'bo')

    for i in solution3:
        plt.axvline(x=i.v, color='b')

    print("Motion-based ARC")
    constraint4 = ArcLength(InvolvedTrussProblemMotionBased())
    solution_method4 = IterativeSolver(constraint4)
    solver4 = IncrementalSolver(solution_method4)
    solution4, tries4 = solver4(Point(u=np.zeros(1), v=np.zeros(1), f=np.zeros(1), p=np.zeros(1)))

    for a in tries4:
        plt.plot([i.v for i in a], [-i.p for i in a], 'ko', alpha=0.1)
        plt.plot([i.u for i in a], [-i.p for i in a], 'ko', alpha=0.1)

    a = [i.v for i in solution4]
    c = [i.u for i in solution4]

    b = [-i.p for i in solution4]
    plt.plot(a, b, 'co')
    plt.plot(c, [-i.p for i in solution4], 'co')




    plt.gca().axis('equal')
    plt.show()
