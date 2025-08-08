import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join(script_dir, '../..')))

from matplotlib import pyplot as plt
import numpy as np

from examples.springable_curves.structure_from_springable import StructureFromSpringableModelFile
from utils import Problem, Plotter
from core import IterativeSolver, IncrementalSolver
from criteria import EigenvalueChangeTermination, termination_default
from dynamics import DynamicsSolver

nlf = StructureFromSpringableModelFile("../springable_curves/csv_files/von_mises_multi_truss.csv")
problem = Problem(nlf, ixf=[2, 4], ixp=[0, 1, 3, 5], ff=np.array([0.0, 1.0]), qp=np.array([0, 0, 0, 0]))
solver = IterativeSolver(problem)

load = termination_default(1.0)
criterion = load | EigenvalueChangeTermination()

stepper = IncrementalSolver(solver, terminated=criterion, reset=False)
stepper.controller.max = 0.1

while not load.exceed: stepper()

plotter = Plotter()
for _, step in enumerate(stepper.history):
    plotter(step.solutions, 2, 4)
    plotter(step.solutions, 4, 4)

### DYNAMICS
pc = stepper.history[0].solutions[-1]
dynsolver = DynamicsSolver(problem)
sol = dynsolver(pc, m=1.0, v0=1.0)

# plot dof 2
plt.plot(sol.y[0], dynsolver.f0[1] * np.ones_like(sol.t), 'mo--')
plt.axvline(sol.y[0][-1])

# plot dof 4
plt.plot(sol.y[1], dynsolver.f0[1] * np.ones_like(sol.t), 'yo--')
plt.axvline(sol.y[1][-1])


plt.show()
