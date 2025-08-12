from math import sqrt

import numpy as np
from matplotlib import pyplot as plt

from constraints import GeneralizedArcLength, NewtonRaphson
from core import IncrementalSolver, IterativeSolver
from criteria import termination_default, EigenvalueChangeTermination
from dynamics import DynamicsSolver
from spring import SpringK
from utils import Problem, Point


def plot_residual(problem, p):
    point = 1.0 * p
    y = np.linspace(2, -2, 30)
    k = np.linspace(3.5, 0, 30)
    r = np.zeros((len(k), len(y)), dtype=float)

    for iy, yname in enumerate(y):
        for ik, kname in enumerate(k):
            point.q[-1] = kname
            point.q[3] = yname
            r[iy, ik] = np.linalg.norm(problem.rf(point))

    k, y = np.meshgrid(k, y)
    plt.contourf(k, y, r, 400, cmap='jet')


spring = Problem(SpringK(sqrt(2)), ixf=[3], ixp=[0, 1, 2, 4],
                 ff=np.zeros(1), qp=np.array([0, 0, 0, -2.5]))

p0 = Point(np.array([0, 0, 1, 1, 3]), np.array([0, 0, 0, -0.2, 0]))

plot_residual(spring, p0)  # plot residual to see disconnected branches

# solve for equilibrium using NR
solver = IterativeSolver(spring, NewtonRaphson())

dp0 = solver([p0])[0]
p0 = p0 + dp0

stepper = IncrementalSolver(solver)

# solve using NR
out_NR = stepper(p0)

# solve using GAL
solver.constraint = GeneralizedArcLength()

out_GAL = stepper(p0)

# solve until critical point using GAL, then use dynamics, then continue
load = termination_default()
criterion = load | EigenvalueChangeTermination()

out_GAL2 = stepper(p0, terminated=criterion)

pc = stepper.history[-1].solutions[-1]  # get first critical point
dynsolver = DynamicsSolver(spring)  # setup dynamics solver

out_dyn = dynsolver(pc, m=1.0, v0=-1.0)

out_GAL3 = stepper(out_dyn[-1], terminated=termination_default(0.4))

# postproccessing
plt.plot([i.q[-1] for i in out_NR.solutions], [i.q[3] for i in out_NR.solutions], 'ro-')
plt.plot([i.q[-1] for i in out_GAL.solutions], [i.q[3] for i in out_GAL.solutions], 'ko-')
plt.plot([i.q[-1] for i in out_GAL2.solutions], [i.q[3] for i in out_GAL2.solutions], 'co-')
plt.plot([i.q[-1] for i in out_dyn], [i.q[3] for i in out_dyn], 'mo-')
plt.plot([i.q[-1] for i in out_GAL3.solutions], [i.q[3] for i in out_GAL3.solutions], 'yo-')

plt.xlim([0, 3.5])
plt.ylim([-2, 2])

plt.show()
