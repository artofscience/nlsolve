import numpy as np
from matplotlib import pyplot as plt

from criteria import residual_norm
from decision_criteria import EigenvalueChangeTermination, LoadTermination
from structure_from_curve import StructureFromCurve
from utils import Problem, plotter
from core import IterativeSolver, IncrementalSolver
from constraints import GeneralizedArcLength
from controllers import Adaptive


nlf = StructureFromCurve("csv_files/jumper.csv")
problem = Problem(nlf, ixp=[0], ixf=[1], ff=np.array([0]), qp=np.array([3]))
solver = IterativeSolver(problem, GeneralizedArcLength(), residual_norm(1e-10))
controller = Adaptive(value=0.1, decr=0.1, incr=1.5, min=0.0001, max=0.2)
stepper = IncrementalSolver(solver, controller,
                            terminated=EigenvalueChangeTermination(),
                            controller_reset=False,
                            history_dependence=True)

stepper()
stepper.step()
stepper.step(terminated=LoadTermination())

h = stepper.history
plotter(h[0].solutions, 0, 0, 'ko-')
plotter(h[1].solutions, 0, 0, 'bo-')
plotter(h[2].solutions, 0, 0, 'ro-')
plt.show()