from math import pi

import numpy as np
from matplotlib import pyplot as plt

from constraints import GeneralizedArcLength, NewtonRaphson
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from decision_criteria import EigenvalueTermination, LoadTermination
from examples.inclined_truss_snapback import InclinedTrussSnapback
from utils import Structure, Point

problem = Structure(InclinedTrussSnapback(theta0=pi / 3), ixf=[0, 1], ff=np.array([0, 0.5]))
controller = Adaptive(0.01, max=0.5, incr=1.2, decr=0.1, min=0.0001)
p0 = Point(qf=np.array([0, 0]), ff=np.array([0, 0]))
solver = IterativeSolver(problem)
stepper = IncrementalSolver(solver, p0, controller)

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

plt.plot([i.qf[0] for i in solution0], [i.ff[1] for i in solution0], 'ko--')
plt.plot([i.qf[1] for i in solution0], [i.ff[1] for i in solution0], 'bo--')

plt.plot([i.qf[0] for i in solution1], [i.ff[1] for i in solution1], 'ko-')
plt.plot([i.qf[1] for i in solution1], [i.ff[1] for i in solution1], 'bo-')

plt.plot([i.qf[0] for i in solution2], [i.ff[1] for i in solution2], 'ro-')
plt.plot([i.qf[1] for i in solution2], [i.ff[1] for i in solution2], 'go-')

plt.plot([i.qf[0] for i in solution3], [i.ff[1] for i in solution3], 'co--')
plt.plot([i.qf[1] for i in solution3], [i.ff[1] for i in solution3], 'yo--')

### END POTTING

plt.show()
