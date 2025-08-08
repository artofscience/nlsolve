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
from matplotlib.pyplot import figure

nlf = StructureFromSpringableModelFile("../springable_curves/csv_files/von_mises_multi_truss.csv")
# problem = Problem(nlf, ixf=[2, 4], ixp=[0, 1, 3, 5], ff=np.array([0.0, 1.0]), qp=np.array([0, 0, 0, 0])) # force load
problem = Problem(nlf, ixf=[2], ixp=[0, 1, 3, 4, 5], ff=np.array([0.0]), qp=np.array([0, 0, 0, 1, 0]))
solver = IterativeSolver(problem)

load = termination_default(5.0)
criterion = load | EigenvalueChangeTermination()

stepper = IncrementalSolver(solver, terminated=criterion, reset=False)

while not load.exceed: stepper()

plotter = Plotter()
for _, step in enumerate(stepper.history):
    plotter(step.solutions, 2, 4)
    plotter(step.solutions, 4, 4)

### DYNAMICS
pc = stepper.history[0].solutions[-1]
dynsolver = DynamicsSolver(problem)
sol = dynsolver(pc, m=0.1, v0=-1.0)

### POST-PROCESSING
state = np.zeros((problem.n, len(sol.t)))
state[problem.ixf, :] = sol.y[0, :]
state[problem.ixp, :] = pc.q[problem.ixp, np.newaxis]

force = np.zeros((problem.n, len(sol.t)))
for i in range(0, len(sol.t)):
    force[:, i] = problem.nlf.force(state[:, i])

reaction_force = force[problem.ixp[3], :]

plt.plot(pc.q[problem.ixp[3]] * np.ones_like(reaction_force), reaction_force, 'mo--')
plt.axhline(reaction_force[-1])
figure()

plt.plot(sol.t, sol.y[0], 'ko--')
plt.plot(sol.t, reaction_force)
plt.show()
