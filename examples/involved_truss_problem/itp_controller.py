from involved_truss_problem import InvolvedTrussProblemLoadBased
import numpy as np
from constraints import GeneralizedArcLength
from core import IncrementalSolver, IterativeSolver, Point
from matplotlib import pyplot as plt
import logging
from controllers import Adaptive

plt.gca().set_xlim([-0.5, 6.5])
plt.gca().set_ylim([-1.0, 1.2])
plt.gca().set_aspect('equal')
plt.ion()
plt.show()

constraint = GeneralizedArcLength()
name = constraint.__class__.__name__+ str(constraint.alpha)

solution_method = IterativeSolver(
    nlf = InvolvedTrussProblemLoadBased(),
    constraint = constraint,
    maximum_corrections = 4,
    name = "IterativeSolver " + name,
    logging_level = logging.DEBUG)

solver = IncrementalSolver(
    solution_method = solution_method,
    name = "IncrementalSolver " + name,
    logging_level = logging.INFO,
    maximum_increments= 1000)

controller = Adaptive(value=0.1, incr=1.5, decr=0.5, min=0.01, max=1.0)

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