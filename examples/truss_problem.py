import numpy as np
from math import pi, sin
from solver import IncrementalSolver, IterativeSolver, Structure
from matplotlib import pyplot as plt


class TrussProblemLoadBased(Structure):
    theta0 = pi/3

    def external_load(self):
        return np.array([-1.0], dtype=float)

    def internal_load_free(self, state, alpha):
        return (1 / np.sqrt(1 - 2 * state * sin(self.theta0) + state ** 2) - 1) * (sin(self.theta0) - state)

    def tangent_stiffness_free_free(self, state):
        return np.array([- 1 / (state ** 2 - 2 * sin(self.theta0) * state + 1) ** (1 / 2) + (
                (state - sin(self.theta0)) * (2 * state - 2 * sin(self.theta0))) / (
                         2 * (state ** 2 - 2 * sin(self.theta0) * state + 1) ** (3 / 2)) + 1])


class TrussProblemMotionBased(Structure):
    theta0 = pi/3

    def prescribed_motion(self):
        return np.array([-1.0], dtype=float)

    def internal_load_prescribed(self, state, alpha):
        return (1 / np.sqrt(1 - 2 * alpha * self.prescribed_motion() * sin(self.theta0) + (alpha * self.prescribed_motion()) ** 2) - 1) * (sin(self.theta0) - alpha * self.prescribed_motion())

    def tangent_stiffness_prescribed_prescribed(self, state):
        return np.array([- 1 / (state ** 2 - 2 * sin(self.theta0) * state + 1) ** (1 / 2) + (
                (state - sin(self.theta0)) * (2 * state - 2 * sin(self.theta0))) / (
                         2 * (state ** 2 - 2 * sin(self.theta0) * state + 1) ** (3 / 2)) + 1])


if __name__ == "__main__":
    nonlinear_function = TrussProblemMotionBased()
    solution_method = IterativeSolver(nonlinear_function)
    solver = IncrementalSolver(solution_method)
    solution = solver()

    b = [i.y for i in solution]
    a = [i.x for i in solution]
    plt.plot(a, b, 'o')
    plt.show()