from math import pi, sin

import numpy as np
from matplotlib import pyplot as plt

from constraints import GeneralizedArcLength
from core import IncrementalSolver, IterativeSolver
from criteria import LoadTermination
from utils import Problem, plotter
from controllers import Adaptive
from operator import gt, lt

"""
Analysis of single-DOF inclined truss with snapthrough behaviour.
"""

class InclinedTruss:
    def __init__(self, theta0: float = pi/3):
        self.theta0 = theta0

    def force(self, a):
        return (1 / np.sqrt(1 - 2 * a * sin(self.theta0) + a ** 2) - 1) * (sin(self.theta0) - a)

    def jacobian(self, a):
        return np.array([- 1 / (a ** 2 - 2 * sin(self.theta0) * a + 1) ** (1 / 2) + (
                (a - sin(self.theta0)) * (2 * a - 2 * sin(self.theta0))) / (
                                 2 * (a ** 2 - 2 * sin(self.theta0) * a + 1) ** (3 / 2)) + 1])



if __name__ == "__main__":

    truss = InclinedTruss(pi / 3)
    problem = Problem(truss, ixf=[0], ff=np.array([-1]))
    solver = IterativeSolver(problem, GeneralizedArcLength())
    controller = Adaptive(0.05, max=0.5, incr=1.2, decr=0.2, min=0.001)
    stepper = IncrementalSolver(solver, controller)

    criteria = LoadTermination(gt, 1.0, 0.01) or LoadTermination(lt, -1.0, 0.01)

    out = stepper(terminated=criteria)
    plotter(out.solutions, 0, 0)
    plt.show()
