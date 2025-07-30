from math import pi, sin

import numpy as np
from matplotlib import pyplot as plt

from constraints import GeneralizedArcLength
from core import IncrementalSolver, IterativeSolver
from utils import Problem, Point
from controllers import Adaptive
from criteria import residual_norm

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

    # first create a truss with angle theta
    truss = InclinedTruss(pi / 3)

    # add boundary and loading conditions
    # in this case we indicate the first index if free and corresponding load magnitude is 1
    # it will find out by itself that this is the only DOF
    problem = Problem(truss, ixf=[0], ff=np.array([1]))

    # create an iterative solver with some constraint and convergence criterium
    solver = IterativeSolver(problem, GeneralizedArcLength(), residual_norm(1e-10))

    # create initial point and provide initial position and load
    p0 = Point(qf=np.zeros(1), ff=np.zeros(1))

    # create a controller that defines the adaptive stepping
    controller = Adaptive(0.05, max=0.5, incr=1.2, decr=0.2, min=0.001)

    # run the stepper given initial point p0 and controller
    stepper = IncrementalSolver(solver)
    solution, _ = stepper(p0, controller)

    # plot position vs force
    plt.plot([i.q for i in solution], [i.f for i in solution], 'ko-')

    plt.show()
