import numpy as np
from matplotlib import pyplot as plt

from core import IncrementalSolver, IterativeSolver
from criteria import termination_default, EigenvalueChangeTermination
from dynamics import DynamicsSolver
from examples.inclined_truss_snapback import InclinedTrussSnapback
from utils import Problem

truss = InclinedTrussSnapback()

problem = Problem(truss, ixf=[0, 1], ff=np.array([0, 1.0]))

solver = IterativeSolver(problem)

load = termination_default(0.5)
criterion = load | EigenvalueChangeTermination()

stepper = IncrementalSolver(solver, terminated=criterion, reset=False)
stepper.controller.max = 1.0

while not load.exceed: stepper()

for _, step in enumerate(stepper.history):
    plt.plot([i.q[0] for i in step.solutions], [i.f[1] for i in step.solutions], 'ko-')
    plt.plot([i.q[1] for i in step.solutions], [i.f[1] for i in step.solutions], 'bo-')

pc = stepper.history[0].solutions[-1]
dynsolver = DynamicsSolver(problem)
sol = dynsolver(pc, alpha=1.0)

plt.plot(sol.y[0], dynsolver.f0[1] * np.ones_like(sol.t), 'mo--')
plt.plot(sol.y[1], dynsolver.f0[1] * np.ones_like(sol.t), 'mo--')

plt.show()
