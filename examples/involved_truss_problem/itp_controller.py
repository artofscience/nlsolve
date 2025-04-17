
from involved_truss_problem import InvolvedTrussProblemLoadBased
import numpy as np
from constraints import GeneralizedArcLength
from core import IncrementalSolver, IterativeSolver
from criteria import CriterionX, CriterionXH, CriterionY, CriterionYH
from utils import Point
from matplotlib import pyplot as plt
import logging
from controllers import Adaptive
from operator import lt, le



plt.gca().set_xlim([-0.5, 6.5])
plt.gca().set_ylim([-1.0, 1.2])
plt.gca().set_aspect('equal')
plt.ion()
plt.show()

constraint = GeneralizedArcLength(
    name = "Constraint",
    logging_level = logging.DEBUG)

problem = InvolvedTrussProblemLoadBased()


residual_norm = CriterionX(
    lambda x, y: np.linalg.norm(x.r(y)), le, 1e-9,
    name="RN", logging_level=logging.DEBUG)

motion_difference = CriterionXH(
    lambda x, y, z: np.linalg.norm(y.q - z.q), lt, 1e-9,
    name="MDN", logging_level=logging.DEBUG)

absolute_load = CriterionY(
    lambda x: abs(x), le, 1e-9,
    name="AL", logging_level=logging.DEBUG)

divergence_criterium = CriterionYH(
    lambda x, y: abs(y) - abs(x), lt, 0.0,
    name="DAL", logging_level=logging.DEBUG)

convergence_criterium = residual_norm & motion_difference & absolute_load
convergence_criterium.logger.name = "ALL"

solution_method = IterativeSolver(
    nlf = problem,
    constraint = constraint,
    converged = convergence_criterium,
    diverged = divergence_criterium,
    maximum_corrections = 100,
    name = "Solver",
    logging_level = logging.DEBUG)

solver = IncrementalSolver(
    solution_method = solution_method,
    name = "Stepper",
    logging_level = logging.DEBUG,
    maximum_increments= 100)

controller = Adaptive(value = 0.1,
                      name = "Controller",
                      logging_level = logging.DEBUG,
                      incr = 1.5, decr = 0.25,
                      min = 0.001, max = 1.0)

initial_point = Point(qf=np.zeros(2), ff=np.zeros(2))
solution, tries = solver(initial_point, controller)

for a in tries:
    plt.plot([i.qf[1] for i in a], [i.ff[1] for i in a], 'ko', alpha=0.1)
    plt.plot([i.qf[0] for i in a], [i.ff[1] for i in a], 'ko', alpha=0.1)

external_load = [i.ff[1] for i in solution]
free_motion_dof1 = [i.qf[1] for i in solution]
free_motion_dof0 = [i.qf[0] for i in solution]
plt.plot(free_motion_dof0, external_load, 'o', alpha=0.5)
plt.plot(free_motion_dof1, external_load, 'o', alpha=0.5)

plt.draw()
plt.pause(12)