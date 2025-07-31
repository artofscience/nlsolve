import numpy as np
from matplotlib import pyplot as plt

from decision_criteria import EigenvalueChangeTermination
from structure_from_curve import StructureFromCurve
from utils import Problem, plotter
from core import IterativeSolver, IncrementalSolver
from constraints import GeneralizedArcLength
from controllers import Adaptive


nlf = StructureFromCurve("csv_files/jumper.csv")
problem = Problem(nlf, ixf=[0, 1], ff=np.array([1, 0]))
solver = IterativeSolver(problem, GeneralizedArcLength())
controller = Adaptive(value=0.3, decr=0.5, incr=1.5)
stepper = IncrementalSolver(solver, controller, controller_reset=False)

out = stepper(terminated=EigenvalueChangeTermination())
plotter(out.solutions, 0, 0, 'ko-')

solver.constraint.direction = False
out = stepper(out.solutions[-1])
plotter(out.solutions, 0, 0, 'bo-')
plt.show()
