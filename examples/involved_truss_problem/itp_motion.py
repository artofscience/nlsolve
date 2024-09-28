from involved_truss_problem import InvolvedTrussProblemMotionBased
import numpy as np
from constraints import NewtonRaphson, ArcLength, GeneralizedArcLength, NewtonRaphsonByArcLength
from core import IncrementalSolver, IterativeSolver
from utils import Point
from matplotlib import pyplot as plt
import logging

constraints = [NewtonRaphson(), NewtonRaphsonByArcLength(), GeneralizedArcLength(alpha=0), ArcLength(), GeneralizedArcLength(alpha=1)]

plt.gca().set_xlim([-0.5, 4.5])
plt.gca().set_ylim([-0.7, 0.7])
plt.gca().set_aspect('equal')
plt.ion()
plt.show()

for constraint in constraints:

    name = constraint.__class__.__name__+(str(constraint.alpha) if hasattr(constraint, 'alpha') else "")

    solution_method = IterativeSolver(
        InvolvedTrussProblemMotionBased(),
        constraint,
        maximum_corrections=100,
        name="IterativeSolver "+name,
        logging_level=logging.DEBUG)

    solver = IncrementalSolver(
        solution_method,
        name="IncrementalSolver "+name,
        logging_level=logging.INFO)

    initial_point = Point(uf=np.zeros(1), up=np.zeros(1), ff=np.zeros(1), fp=np.zeros(1))
    solution, tries = solver(initial_point)

    for a in tries:
        plt.plot([i.up for i in a], [i.fp for i in a], 'ko', alpha=0.1)
        plt.plot([i.uf for i in a], [i.fp for i in a], 'ko', alpha=0.1)

    plt.plot([i.up for i in solution], [i.fp for i in solution], 'o', alpha=0.5)
    plt.plot([i.uf for i in solution], [i.fp for i in solution], 'o', alpha=0.5)

    if constraint.__class__.__name__ == "NewtonRaphson":
        for i in solution:
            plt.axvline(i.up)

    plt.draw()
    print("Pause...")
    plt.pause(2)
