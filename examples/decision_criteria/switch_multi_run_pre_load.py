from math import pi

import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

from constraints import GeneralizedArcLength, NewtonRaphson
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from decision_criteria import EigenvalueTermination, LoadTermination, EigenvalueChangeTermination
from examples.inclined_truss_snapback import InclinedTrussSnapback
from utils import Structure, Point

problem = Structure(InclinedTrussSnapback(theta0=pi / 3), ixf=[0, 1], ff=np.array([0, 0.5]))
controller = Adaptive(0.05, incr=1.3, decr=0.1, min=0.00001)
p0 = Point(qf=np.array([0, 0]), ff=np.array([0, 0]))
constraint = GeneralizedArcLength()

solver = IterativeSolver(problem, constraint)

stepper = IncrementalSolver(solver, p0, controller, maximum_increments=100)

solution0 = stepper(terminated=EigenvalueChangeTermination())[0]

plt.plot([i.qf[0] for i in solution0], [i.ff[1] for i in solution0], 'ko--',  markersize=10)
plt.plot([i.qf[1] for i in solution0], [i.ff[1] for i in solution0], 'bo--', markersize=10)

pswitch = deepcopy(solution0[-1])
pswitch.y = 0.0

constraint.direction = False
solution1 = stepper(pswitch, terminated=EigenvalueChangeTermination())[0]

plt.plot([i.qf[0] for i in solution1], [i.ff[1] for i in solution1], 'yo--')
plt.plot([i.qf[1] for i in solution1], [i.ff[1] for i in solution1], 'go--')

pswitch = deepcopy(solution1[-1])
pswitch.y = 0.0

constraint.direction = True
solution2 = stepper(pswitch, terminated=LoadTermination())[0]

plt.plot([i.qf[0] for i in solution2], [i.ff[1] for i in solution2], 'ro--')
plt.plot([i.qf[1] for i in solution2], [i.ff[1] for i in solution2], 'co--')


plt.show()
