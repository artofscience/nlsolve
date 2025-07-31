import numpy as np
from matplotlib import pyplot as plt

from decision_criteria import EigenvalueChangeTermination, LoadTermination
from structure_from_curve import StructureFromCurve
from utils import Problem, plotter
from core import IterativeSolver, IncrementalSolver
from constraints import GeneralizedArcLength
from controllers import Adaptive


nlf = StructureFromCurve("csv_files/jumper.csv")
problem = Problem(nlf, ixf=[0 , 1], ff=np.array([3, 0]))
solver = IterativeSolver(problem, GeneralizedArcLength())
controller = Adaptive(value=0.1, decr=0.1, incr=1.5, min=0.0001, max=0.2)
stepper = IncrementalSolver(solver, controller,
                            terminated=EigenvalueChangeTermination(),
                            controller_reset=False)

out = stepper()
plotter(out.solutions, 0, 0, 'ko-')

solver.constraint.direction = False
out = stepper(out.solutions[-1])
plotter(out.solutions, 0, 0, 'bo-')

solver.constraint.direction = True
out = stepper(out.solutions[-1])
plotter(out.solutions, 0, 0, 'ro-')

solver.constraint.direction = False
out = stepper(out.solutions[-1])
plotter(out.solutions, 0, 0, 'co-')

solver.constraint.direction = True
out = stepper(out.solutions[-1], terminated=LoadTermination())
plotter(out.solutions, 0, 0, 'go-')

plt.show()
