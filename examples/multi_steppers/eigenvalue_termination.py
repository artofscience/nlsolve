import numpy as np
from matplotlib import pyplot as plt

from constraints import GeneralizedArcLength
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from criteria import residual_norm, EigenvalueTermination
from examples.inclined_truss_snapback import InclinedTrussSnapback
from utils import Problem, plotter
from operator import lt

problem = Problem(InclinedTrussSnapback(), ixf=[0, 1], ff=np.array([0, 0.5]))

solver = IterativeSolver(problem, GeneralizedArcLength(), residual_norm(1e-6))

controller = Adaptive(0.5, max=0.5, incr=1.2, decr=0.1, min=0.0001)

stepper = IncrementalSolver(solver, controller=controller)

# stepper.controller_reset = False
out = stepper()

plotter(out.solutions, 0, 1, 'ko-')
plotter(out.solutions, 1, 1, 'bo-')

# then solve for eigenvalue termination
decision = EigenvalueTermination(lt, -0.2, 0.01)

out = stepper(terminated=decision)

plotter(out.solutions, 0, 1, 'ro-')
plotter(out.solutions, 1, 1, 'yo-')

plt.show()
