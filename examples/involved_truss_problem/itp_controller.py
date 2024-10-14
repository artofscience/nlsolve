
from involved_truss_problem import InvolvedTrussProblemLoadBased
import numpy as np
from constraints import GeneralizedArcLength
from core import IncrementalSolver, IterativeSolver
from criteria import ConvergenceCriterion
from utils import Point
from matplotlib import pyplot as plt
import logging
from controllers import Adaptive
import operator



plt.gca().set_xlim([-0.5, 6.5])
plt.gca().set_ylim([-1.0, 1.2])
plt.gca().set_aspect('equal')
plt.ion()
plt.show()

constraint = GeneralizedArcLength(
    name = "Constraint",
    logging_level = logging.DEBUG)

problem = InvolvedTrussProblemLoadBased()

residual_norm = ConvergenceCriterion(
    lambda x, y: np.linalg.norm(x.r(y)), operator.lt, 1e-9,
    name="RN", logging_level=logging.DEBUG)

max_abs_residual_free_dofs = ConvergenceCriterion(
    lambda x, y: np.max(np.abs((x.rf(y)))), operator.le, 1e-9,
    name="MARF", logging_level=logging.DEBUG)

criterium = residual_norm & max_abs_residual_free_dofs
criterium.logger.name = "RN&MARF"

solution_method = IterativeSolver(
    nlf = problem,
    constraint = constraint,
    converged = criterium,
    maximum_corrections = 4,
    name = "Solver",
    logging_level = logging.DEBUG)

solver = IncrementalSolver(
    solution_method = solution_method,
    name = "Stepper",
    logging_level = logging.DEBUG,
    maximum_increments= 1000)

controller = Adaptive(value = 0.1,
                      name = "Controller",
                      logging_level = logging.DEBUG,
                      incr = 1.5, decr = 0.5,
                      min = 0.01, max = 1.0)

initial_point = Point(uf=np.zeros(2), ff=np.zeros(2))
solution, tries = solver(initial_point, controller)

for a in tries:
    plt.plot([i.uf[1] for i in a], [i.ff[1] for i in a], 'ko', alpha=0.1)
    plt.plot([i.uf[0] for i in a], [i.ff[1] for i in a], 'ko', alpha=0.1)

external_load = [i.ff[1] for i in solution]
free_motion_dof1 = [i.uf[1] for i in solution]
free_motion_dof0 = [i.uf[0] for i in solution]
plt.plot(free_motion_dof0, external_load, 'o', alpha=0.5)
plt.plot(free_motion_dof1, external_load, 'o', alpha=0.5)

plt.draw()
plt.pause(6)