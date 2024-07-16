import numpy as np
from math import pi, sin
from solver import IncrementalSolver, IterativeSolver
from matplotlib import pyplot as plt


class InvolvedTrussProblemLoadBased:
    w = 0.1
    theta0 = pi/3

    @staticmethod
    def external_load():
        return np.array([0, -1.0], dtype=float)

    def internal_load_free(self, state):
        f1 = (1 / np.sqrt(1 - 2 * state[0] * sin(self.theta0) + state[0] ** 2) - 1) * (sin(self.theta0) - state[0]) - self.w * (state[1] - state[0])
        return np.array([f1, self.w * (state[1] - state[0])])

    def residual_free(self, state, alpha):
        return self.internal_load_free(state) + self.external_load(alpha)

    def tangent_stiffness_free_free(self, state):
        a1 = state[0]
        df1da1 = self.w - 1 / (a1 ** 2 - 2 * sin(self.theta0) * a1 + 1) ** (1 / 2) + (
                (a1 - sin(self.theta0)) * (2 * a1 - 2 * sin(self.theta0))) / (
                         2 * (a1 ** 2 - 2 * sin(self.theta0) * a1 + 1) ** (3 / 2)) + 1
        return np.array([[df1da1, -self.w], [-self.w, self.w]], dtype=float)


class InvolvedTrussProblemMotionBased(InvolvedTrussProblemLoadBased):
    def external_load(self, alpha):
        return 0.0

    def prescribed_motion(self):
        return 1.0

    def internal_load_prescribed(self, state, alpha):
        return self.w * (alpha * self.prescribed_motion() - state)

    def internal_load_free(self, state, alpha):
        return (1 / np.sqrt(1 - 2 * state[0] * sin(self.theta0) + state[0] ** 2) - 1) * (sin(self.theta0) - state[0]) - self.w * (alpha * self.prescribed_motion() - state[0])

    def residual_free(self, state, alpha):
        return self.internal_load_free(state) + self.external_load(alpha)

    def tangent_stiffness_free_free(self, state):
        a1 = state[0]
        return self.w - 1 / (a1 ** 2 - 2 * sin(self.theta0) * a1 + 1) ** (1 / 2) + (
                (a1 - sin(self.theta0)) * (2 * a1 - 2 * sin(self.theta0))) / (
                         2 * (a1 ** 2 - 2 * sin(self.theta0) * a1 + 1) ** (3 / 2)) + 1




if __name__ == "__main__":
    nonlinear_function = InvolvedTrussProblemLoadBased()
    solution_method = IterativeSolver(nonlinear_function)
    solver = IncrementalSolver(solution_method)
    solution = solver()

    b = [i.y for i in solution]
    for j in [0, 1]:
        a = [i.x[j] for i in solution]
        plt.plot(a, b, 'o')
    plt.show()

    print(b)
