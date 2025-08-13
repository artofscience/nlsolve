import numpy as np
from matplotlib import pyplot as plt

from constraints import NewtonRaphson
from core import IncrementalSolver, IterativeSolver
from spring import SpringL0
from utils import Problem, Point

# setup horizontal spring problem free in y-movement of second node
# with driven rest length from 0 to 2
spring = Problem(SpringL0(), ixf = [3], ixp = [0, 1, 2, 4],
                 ff = np.zeros(1), qp = np.array([0, 0, 0, 2.0]))

solver = IterativeSolver(spring, NewtonRaphson())
stepper = IncrementalSolver(solver)
p0 = Point(q=np.array([0, 0, 1, 0, 0]), f=np.zeros(5))

for j in [-1e-6, 0, 1e-6]:
    p0.f[3] = j
    dp0 = solver([p0])[0]
    out = stepper(p0 + dp0)
    solution = out.solutions
    tries = out.tries

    plt.plot([i.q[4] for i in solution], [i.q[3] for i in solution], 'ro-')
    plt.plot([i.q[4] for i in solution], [i.f[-1] for i in solution], 'bo-')

    for a in tries:
        plt.plot([i.q[4] for i in a], [i.q[3] for i in a], 'ro--', alpha=0.1)
        plt.plot([i.q[4] for i in a], [i.f[-1] for i in a], 'bo--', alpha=0.1)

# CHECK REACTION FORCE VALUE
plt.show()
