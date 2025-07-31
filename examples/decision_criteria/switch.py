from math import pi

import numpy as np
from matplotlib import pyplot as plt

from constraints import GeneralizedArcLength, NewtonRaphson
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from decision_criteria import EigenvalueTermination, LoadTermination
from examples.inclined_truss_snapback import InclinedTrussSnapback
from utils import Problem, Point, plotter

problem = Problem(InclinedTrussSnapback(theta0=pi / 3), ixf=[0, 1], ff=np.array([0, 0.5]))
controller = Adaptive(0.01, max=0.5, incr=1.2, decr=0.1, min=0.0001)
solver = IterativeSolver(problem)
stepper = IncrementalSolver(solver, controller)

# STEP 0: NR WITH LOAD TERMINATION
out0 = stepper()

# STEP 1: ARCLENGTH WITH LOAD TERMINATION
out1 = stepper(constraint=GeneralizedArcLength())

# STEP 2: ARCLENGTH WITH EIGENVALUE TERMINATION
out2 = stepper(terminated=EigenvalueTermination(-0.4, 0.01))

# STEP 3: NR WITH LOAD TERMINATION
# STARTING FROM SOLUTION2
out3 = stepper(out2.solutions[-1], constraint=NewtonRaphson(), terminated=LoadTermination(1.0, 0.1))

### PLOTTING
plotter(out0.solutions, 0, 1, 'ko--')
plotter(out0.solutions, 1, 1, 'ko--')

plotter(out1.solutions, 0, 1, 'ko-')
plotter(out1.solutions, 1, 1, 'ko-')

plotter(out2.solutions, 0, 1, 'ro-')
plotter(out2.solutions, 1, 1, 'go-')

plotter(out3.solutions, 0, 1, 'co--')
plotter(out3.solutions, 1, 1, 'yo--')

### END POTTING

plt.show()
