import numpy as np
from solver import Structure, IterativeSolver, IncrementalSolver
from matplotlib import pyplot as plt

class TwoSprings(Structure):
    l0 = 1
    k1 = 1
    k2 = 1

    def external_load(self):
        return np.array([0, 0.6])

    def internal_load_free(self, p):
        l1 = self.l0 + p.u[0]
        l2 = self.l0 + p.u[1] - p.u[0]
        e1 = 0.5 * (l1 / self.l0)**2 - 0.5
        e2 = 0.5 * (l2 / self.l0)**2 - 0.5
        return np.array([e1 + e2, -e2])

    def tangent_stiffness_free_free(self, p):
        l1 = self.l0 + p.u[0]
        l2 = self.l0 + p.u[1] - p.u[0]
        e1 = 0.5 * (l1 / self.l0)**2 - 0.5
        e2 = 0.5 * (l2 / self.l0)**2 - 0.5
        return np.array([[l1 / self.l0**2 - l2 / self.l0**2, l2 / self.l0**2],
                         [l2 / self.l0**2, -l2 / self.l0**2]])

    # def internal_load_free(self, p):
    #     a = p.u[0] - p.u[1]
    #     return np.array([self.k1l * p.u[0] ** 2 + self.k2l * a ** 2 + self.k1 * p.u[0] + self.k2 * a,
    #                      -self.k2l * a ** 2 - self.k2 * a])

    # def tangent_stiffness_free_free(self, p):
    #     a = p.u[0] - p.u[1]
    #     return 2 * np.array([[self.k1l * p.u[0] + self.k2l * a + self.k1 + self.k2, -self.k2l * a - self.k2],
    #                          [-self.k2l * a - self.k2, self.k2l * p.u[1] + self.k2]])


if __name__ == "__main__":
    print("Load-based NR")
    solution_method1 = IterativeSolver(TwoSprings())
    solver1 = IncrementalSolver(solution_method1, alpha=1.0)
    solution1, tries1 = solver1()

    # for a in tries1:
    #     plt.plot([i.u[1] for i in a], [-i.f[1] for i in a], 'ko', alpha=0.1)
    #     plt.plot([i.u[0] for i in a], [-i.f[1] for i in a], 'ko', alpha=0.1)
    b = [i.f[1] for i in solution1]
    a = [i.u[1] for i in solution1]
    c = [i.u[0] for i in solution1]
    plt.plot(a, b, 'ro')
    # plt.plot(c, b, 'ro')
    # for i in solution1:
    #     plt.axhline(y=i.f[1], color='r')

    plt.gca().axis('equal')
    plt.show()
