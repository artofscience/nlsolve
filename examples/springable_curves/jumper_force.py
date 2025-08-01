import numpy as np
from matplotlib import pyplot as plt

from decision_criteria import EigenvalueChangeTermination
from criteria import LoadTermination
from structure_from_springable import StructureFromCurve
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
                            reset=False)

stepper()
stepper.step()
stepper.step()
stepper.step()
stepper.step(terminated=LoadTermination())

h = stepper.history
plotter(h[0].solutions, 0, 0, 'ko-')
plotter(h[1].solutions, 0, 0, 'bo-')
plotter(h[2].solutions, 0, 0, 'ro-')
plotter(h[3].solutions, 0, 0, 'co-')
plotter(h[4].solutions, 0, 0, 'go-')

plt.show()