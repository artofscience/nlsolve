import numpy as np
from matplotlib import pyplot as plt

from constraints import NewtonRaphson, NewtonRaphsonByArcLength, ArcLength, GeneralizedArcLength
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from inclined_truss_snapback import InclinedTrussSnapback
from utils import Problem

problem = Problem(InclinedTrussSnapback(), ixf=[0, 1], ff=np.array([0, 0.5]))
controller = Adaptive(0.05, max=0.5, incr=1.2, decr=0.1, min=0.0001)

# loop over constraints
constraints = [NewtonRaphson(), NewtonRaphsonByArcLength(), GeneralizedArcLength(alpha=0.0), ArcLength(),
               GeneralizedArcLength(alpha=1.0, beta=0.0)]

solver = IterativeSolver(problem)
stepper = IncrementalSolver(solver, controller)

fig, ax = plt.subplots(1, 5)
for count, constraint in enumerate(constraints):
    solver.constraint = constraint
    stepper()

for count, out in enumerate(stepper.history):
    ax[count].plot([i.q[0] for i in out.solutions], [i for i in out.time], 'bo-')
    ax[count].plot([i.q[1] for i in out.solutions], [i for i in out.time], 'ko-')

plt.show()
