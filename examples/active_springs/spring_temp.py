import numpy as np
from matplotlib import pyplot as plt

from core import IncrementalSolver, IterativeSolver
from spring import SpringT
from utils import Problem, Point
from sympy import Symbol


# dofs = [x0, y0, x1, y1, T]
ixp = [0, 1, 2, 3, 4]

# setup loading conditions
qp = np.zeros(5)
qp[4] = 1.0

# setup temperature-sensitive spring
"""Assume here both k and l0 are linearly increasing with T
k = 1 + lambda * T
l0 = 1 + lambda * T"""
T = Symbol("T")
spring = SpringT(l0 = 1 + T, k = 2 - 1.99*T)

# setup problem
structure = Problem(spring, ixp=ixp, qp=qp)

# setup solver
solver = IterativeSolver(structure)

# initial point
p0 = Point(q=np.array([0, 0, 1, 0, 0]))

# solve for equilibrium given initial point
dp0 = solver([p0])[0]

# setup stepper
steppah = IncrementalSolver(solver)

# solve problem from equilibrium point
out = steppah(p0 + dp0)
solution = out.solutions

# plot temperature and entropy (flux)
plt.plot([i.q[4] for i in solution], [i.q[4] for i in solution], 'ko-')
plt.plot([i.q[4] for i in solution], [i.f[4] for i in solution], 'ko--')

# plot reaction forces on nodes
plt.plot([i.q[4] for i in solution], [i.f[0] for i in solution], 'ro-')
plt.plot([i.q[4] for i in solution], [i.f[2] for i in solution], 'ro--')

plt.plot([i.q[4] for i in solution], [i.f[1] for i in solution], 'go-')
plt.plot([i.q[4] for i in solution], [i.f[3] for i in solution], 'go--')

plt.show()
