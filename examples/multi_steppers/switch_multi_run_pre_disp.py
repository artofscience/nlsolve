from copy import deepcopy
from math import pi
from operator import gt

import numpy as np
from matplotlib import pyplot as plt

from constraints import GeneralizedArcLength
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from criteria import LoadTermination, EigenvalueChangeTermination
from examples.inclined_truss_snapback import InclinedTrussSnapback
from utils import Problem, Plotter

problem = Problem(InclinedTrussSnapback(theta0=pi / 3), ixf=[0], ff=np.array([0]), ixp=[1], qp=np.array([1]))
constraint = GeneralizedArcLength()

solver = IterativeSolver(problem, constraint)

stepper = IncrementalSolver(solver, reset=False)

out = stepper(terminated=EigenvalueChangeTermination())

plotter = Plotter()
plotter(out.solutions, 1, 1)
plotter(out.solutions, 0, 1)

pswitch = deepcopy(out.solutions[-1])

out2 = stepper(pswitch, terminated=EigenvalueChangeTermination())

plotter(out2.solutions, 1, 1)
plotter(out2.solutions, 0, 1)

pswitch = deepcopy(out2.solutions[-1])

out3 = stepper(pswitch, terminated=LoadTermination(gt, 6, 0.01))

plotter(out3.solutions, 1, 1)
plotter(out3.solutions, 0, 1)

plt.show()
