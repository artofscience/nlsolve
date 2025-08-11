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

load = termination_default()
criterion = load | EigenvalueChangeTermination()

stepper = IncrementalSolver(solver, terminated=criterion, reset=False)

while not load.exceed: stepper()

plotter = Plotter()
for _, step in enumerate(stepper.history):
    plotter(step.solutions, 2, 4)
    plotter(step.solutions, 4, 4)

pc = stepper.history[0].solutions[-1]
dynsolver = DynamicsSolver(problem)
out = dynsolver(pc, m=2.0, v0=4.0)

plotter = Plotter()
plotter(out, 4, 4)
plotter(out, 2, 4)

pc2 = dynsolver.load_based_offset(pc)
out = dynsolver(pc2)

plotter(out, 4, 4)
plotter(out, 2,4)

plt.show()
