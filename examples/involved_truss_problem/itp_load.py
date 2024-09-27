from involved_truss_problem import InvolvedTrussProblemLoadBased
import numpy as np
from constraints import NewtonRaphson, ArcLength, GeneralizedArcLength, NewtonRaphsonByArcLength
from core import IncrementalSolver, IterativeSolver, Point
from matplotlib import pyplot as plt
import logging

constraints = [NewtonRaphson(), NewtonRaphsonByArcLength(), GeneralizedArcLength(alpha=0), ArcLength(), GeneralizedArcLength(alpha=1)]

plt.gca().set_xlim([-0.5, 6.5])
plt.gca().set_ylim([-1.0, 1.2])
plt.gca().set_aspect('equal')
plt.ion()
plt.show()

for constraint in constraints:
    name = constraint.__class__.__name__+(str(constraint.alpha) if hasattr(constraint, 'alpha') else "")

    solution_method = IterativeSolver(
        InvolvedTrussProblemLoadBased(),
        constraint,
        maximum_corrections=100,
        name="IterativeSolver "+name,
        logging_level=logging.DEBUG)

    solver = IncrementalSolver(
        solution_method,
        name="IncrementalSolver "+name,
        logging_level=logging.INFO)

    initial_point = Point(uf=np.zeros(2), ff=np.zeros(2))
    solution, tries = solver(initial_point)

    for a in tries:
        plt.plot([i.uf[1] for i in a], [i.ff[1] for i in a], 'ko', alpha=0.1)
        plt.plot([i.uf[0] for i in a], [i.ff[1] for i in a], 'ko', alpha=0.1)

    external_load = [i.ff[1] for i in solution]
    free_motion_dof1 = [i.uf[1] for i in solution]
    free_motion_dof0 = [i.uf[0] for i in solution]
    plt.plot(free_motion_dof0, external_load, 'o', alpha=0.5)
    plt.plot(free_motion_dof1, external_load, 'o', alpha=0.5)

    if constraint.__class__.__name__ == "NewtonRaphson":
        for i in solution:
            plt.axhline(i.ff[1])

    plt.draw()
    print("Pause...")
    plt.pause(2)
