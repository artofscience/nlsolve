import numpy as np
from matplotlib import pyplot as plt

from core import IterativeSolver, IncrementalSolver
from criteria import termination_default, EigenvalueChangeTermination
from structure_from_springable import StructureFromCurve
from utils import Problem, Plotter

nlf = StructureFromCurve("csv_files/jumper.csv")
problem = Problem(nlf, ixf=[0, 1], ff=np.array([3, 0]))
solver = IterativeSolver(problem)

criterion = termination_default() | EigenvalueChangeTermination()
stepper = IncrementalSolver(solver, terminated=criterion, reset=False)

plotter = Plotter()
while not criterion.left.exceed:
    stepper()
    plotter(stepper.out.solutions, 0, 0)

plt.show()
