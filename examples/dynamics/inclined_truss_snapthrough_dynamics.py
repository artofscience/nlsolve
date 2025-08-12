import numpy as np
from matplotlib import pyplot as plt

from core import IncrementalSolver, IterativeSolver
from criteria import EigenvalueChangeTermination, termination_default
from dynamics import DynamicsSolver
from examples.inclined_truss_snapthrough import InclinedTruss
from utils import Problem, Plotter

problem = Problem(InclinedTruss(), ixf=[0], ff=np.array([1]))
solver = IterativeSolver(problem)

load = termination_default()
criterion = load | EigenvalueChangeTermination()

stepper = IncrementalSolver(solver, terminated=criterion, reset=False)

while not load.exceed: stepper()

plotter = Plotter()
for _, step in enumerate(stepper.history): plotter(step.solutions, 0, 0)

### DYNAMIC ANALYSIS
pc = stepper.history[0].solutions[-1]
dynsolver = DynamicsSolver(problem)

sol = dynsolver(pc)
plt.plot(sol.y[0], dynsolver.f0 * np.ones_like(sol.t), 'm.--', markersize=50)

sol = dynsolver(pc, alpha=1.01)
plt.plot(sol.y[0], dynsolver.f0 * np.ones_like(sol.t), 'yo--')

sol = dynsolver(pc, m=1.0)
plt.plot(sol.y[0], dynsolver.f0 * np.ones_like(sol.t), 'ko--', markersize=20)

sol = dynsolver(pc, m=1.0, alpha=1.02)
plt.plot(sol.y[0], dynsolver.f0 * np.ones_like(sol.t), 'bo--')

sol = dynsolver(pc, m=1.0, v0=1.0)
plt.plot(sol.y[0], dynsolver.f0 * np.ones_like(sol.t), 'go--')

plt.show()
