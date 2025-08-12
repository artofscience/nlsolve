import numpy as np
from matplotlib import pyplot as plt

from constraints import GeneralizedArcLength
from controllers import Adaptive
from core import IterativeSolver, IncrementalSolver
from criteria import EigenvalueChangeTermination, termination_default
from structure_from_springable import StructureFromCurve
from utils import Problem, Plotter

nlf = StructureFromCurve("csv_files/jumper.csv")
problem = Problem(nlf, ixp=[0], ixf=[1], ff=np.array([0]), qp=np.array([3]))
solver = IterativeSolver(problem, GeneralizedArcLength())
controller = Adaptive(value=0.1, decr=0.1, incr=1.5, min=0.0001, max=0.2)

criterion = termination_default() | EigenvalueChangeTermination(0.1)
stepper = IncrementalSolver(solver, controller,
                            terminated=criterion,
                            reset=False)
plotter = Plotter()
while not criterion.left.exceed:
    stepper()
    plotter(stepper.out.solutions, 0, 0)

plt.show()
