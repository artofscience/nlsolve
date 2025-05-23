from matplotlib import pyplot as plt
import numpy as np
from core import IncrementalSolver, IterativeSolver
from utils import Structure, Point
from criteria import residual_norm

from spring_defs import SpringL0


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

spring = Structure(SpringL0(), ixf, ixp, ff, qp)

solver = IterativeSolver(spring, converged=residual_norm(0.01))
stepper = IncrementalSolver(solver)

for j in [-1e-6, 0, 1e-6]:
    point = Point(qp=np.array([0, 0, 1, 0]), ff=np.array([j]))
    solution, tries = stepper(point + solver([point])[0])

    ax1.plot([i.y for i in solution], [i.qf for i in solution], 'ro-')
    ax2.plot([i.y for i in solution], [i.fp[-1] for i in solution], 'bo-')

    for a in tries:
        ax1.plot([i.y for i in a], [i.qf for i in a], 'ro--', alpha=0.1)
        ax2.plot([i.y for i in a], [i.fp[-1] for i in a], 'bo--', alpha=0.1)


# CHECK REACTION FORCE VALUE
plt.show()