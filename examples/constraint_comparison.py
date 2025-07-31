import numpy as np

from inclined_truss_snapback import InclinedTrussSnapback
from constraints import NewtonRaphson, NewtonRaphsonByArcLength, ArcLength, GeneralizedArcLength
from core import IncrementalSolver, IterativeSolver
from utils import Problem, Point, plotter
from controllers import Adaptive
from matplotlib import pyplot as plt

problem = Problem(InclinedTrussSnapback(), ixf=[0, 1], ff=np.array([0, 0.5]))
controller = Adaptive(0.05, max=0.5, incr=1.2, decr=0.1, min=0.0001)

# loop over constraints
constraints = [NewtonRaphson(), NewtonRaphsonByArcLength(), GeneralizedArcLength(alpha=0.0), ArcLength(), GeneralizedArcLength(alpha=1.0, beta=0.0)]

solver = IterativeSolver(problem)
stepper = IncrementalSolver(solver, controller)

fig, ax = plt.subplots(1,5)
for count, constraint  in enumerate(constraints):
    solver.constraint = constraint
    stepper()

for count, out in enumerate(stepper.history):
    ax[count].plot([i.q[0] for i in out.solutions], [i for i in out.time], 'bo-')
    ax[count].plot([i.q[1] for i in out.solutions], [i for i in out.time], 'ko-')

plt.show()
