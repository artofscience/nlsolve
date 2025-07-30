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
solution0 = stepper()[0]

# STEP 1: ARCLENGTH WITH LOAD TERMINATION
solution1 = stepper(constraint=GeneralizedArcLength())[0]

# STEP 2: ARCLENGTH WITH EIGENVALUE TERMINATION
solution2 = stepper(terminated=EigenvalueTermination(-0.4, 0.01))[0]

# STEP 3: NR WITH LOAD TERMINATION
# STARTING FROM SOLUTION2
solution3 = stepper(solution2[-1], constraint=NewtonRaphson(), terminated=LoadTermination(1.0, 0.1))[0]

### PLOTTING
plotter(solution0, 0, 1, 'ko--')
plotter(solution0, 1, 1, 'ko--')

plotter(solution1, 0, 1, 'ko-')
plotter(solution1, 1, 1, 'ko-')

plotter(solution2, 0, 1, 'ro-')
plotter(solution2, 1, 1, 'go-')

plotter(solution3, 0, 1, 'co--')
plotter(solution3, 1, 1, 'yo--')

### END POTTING

plt.show()
