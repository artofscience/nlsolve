from math import pi

import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

from constraints import GeneralizedArcLength
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from decision_criteria import LoadTermination, EigenvalueChangeTermination
from examples.inclined_truss_snapback import InclinedTrussSnapback
from utils import Problem, Point, plotter

problem = Problem(InclinedTrussSnapback(theta0=pi / 3), ixf=[0, 1], ff=np.array([0, 0.5]))
controller = Adaptive(0.05, incr=1.3, decr=0.1, min=0.00001)
constraint = GeneralizedArcLength()

solver = IterativeSolver(problem, constraint)

stepper = IncrementalSolver(solver, controller, maximum_increments=100)

solution0 = stepper(terminated=EigenvalueChangeTermination())

plotter(solution0.solutions, 1, 1, 'ko--')
plotter(solution0.solutions, 0, 1, 'bo--')

pswitch = deepcopy(solution0.solutions[-1])

constraint.direction = False
solution1 = stepper(pswitch, terminated=EigenvalueChangeTermination())


plotter(solution1.solutions, 1, 1, 'yo--')
plotter(solution1.solutions, 0, 1, 'go--')

pswitch = deepcopy(solution1.solutions[-1])

constraint.direction = True
solution2 = stepper(pswitch, terminated=LoadTermination())


plotter(solution2.solutions, 1, 1, 'ro--')
plotter(solution2.solutions, 0, 1, 'co--')


plt.show()