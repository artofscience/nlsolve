from matplotlib import pyplot as plt
import numpy as np
from core import IncrementalSolver, IterativeSolver
from utils import Problem, Point
from criteria import residual_norm

from spring import SpringL0


fig, ax1 = plt.subplots()

ax1.set_xlabel('lambda')
ax1.set_ylabel('Position', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('Load', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

ixf = [3]
ixp = [0, 1, 2, 4]

ff = np.zeros(1)
qp = np.zeros(len(ixp))
qp[-1] = 2.0

spring = Problem(SpringL0(), ixf, ixp, ff, qp)

solver = IterativeSolver(spring, converged=residual_norm(0.01))
stepper = IncrementalSolver(solver)

for j in [-1e-6, 0, 1e-6]:
    point = Point(q=np.array([0, 0, 1, 0, 0]), f=np.array([0, 0, 0, j, 0]))
    solution, tries = stepper(point + solver([point])[0])

    ax1.plot([i.q[4] for i in solution], [i.q[3] for i in solution], 'ro-')
    ax2.plot([i.q[4] for i in solution], [i.f[-1] for i in solution], 'bo-')

    for a in tries:
        ax1.plot([i.q[4] for i in a], [i.q[3] for i in a], 'ro--', alpha=0.1)
        ax2.plot([i.q[4] for i in a], [i.f[-1] for i in a], 'bo--', alpha=0.1)


# CHECK REACTION FORCE VALUE
plt.show()