from math import pi

import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

from constraints import GeneralizedArcLength, NewtonRaphson
from controllers import Adaptive, Controller
from core import IncrementalSolver, IterativeSolver
from decision_criteria import EigenvalueTermination, LoadTermination, EigenvalueChangeTermination
from examples.inclined_truss_snapback import InclinedTrussSnapback
from utils import Problem, Point, plotter

problem = Problem(InclinedTrussSnapback(theta0=pi / 3), ixf=[0], ff=np.array([0]), ixp=[1], qp=np.array([1]))
controller = Adaptive(0.01, incr=1.3, decr=0.1, min=0.00001)
constraint = GeneralizedArcLength()

solver = IterativeSolver(problem, constraint)

stepper = IncrementalSolver(solver, controller, reset=False)


out = stepper(terminated=EigenvalueChangeTermination())

plotter(out.solutions, 1, 1, 'ko--')
plotter(out.solutions, 0, 1, 'bo--')

pswitch = deepcopy(out.solutions[-1])

out2 = stepper(pswitch, terminated=EigenvalueChangeTermination())

plotter(out2.solutions, 1, 1, 'yo--')
plotter(out2.solutions, 0, 1, 'go--')

pswitch = deepcopy(out2.solutions[-1])

out3 = stepper(pswitch, terminated=LoadTermination(6, 0.01))

plotter(out3.solutions, 1, 1, 'ro--')
plotter(out3.solutions, 0, 1, 'co--')


plt.show()
