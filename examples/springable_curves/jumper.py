import numpy as np
from matplotlib import pyplot as plt

from structure_from_curve import StructureFromCurve
from utils import Problem, plotter
from core import IterativeSolver, IncrementalSolver
from constraints import ArcLength
from controllers import Adaptive


nlf = StructureFromCurve("csv_files/jumper.csv")
problem = Problem(nlf, ixf=[0, 1], ff=np.array([3, 0]))
solver = IterativeSolver(problem, ArcLength())
stepper = IncrementalSolver(solver, Adaptive(value=0.3, decr=0.9, incr=1.3))
out = stepper()

plotter(out.solutions, 0, 0)
plt.show()
