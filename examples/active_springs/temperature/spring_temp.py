from matplotlib import pyplot as plt
import numpy as np

from core import IncrementalSolver, IterativeSolver
from utils import Problem, Point

from spring import SpringT

# dofs = [x0, y0, x1, y1, T]
ixp = [0, 1, 2, 3, 4]

# setup loading conditions
qp = np.zeros(5)
qp[4] = 1.0

# setup temperature-sensitive spring
"""Assume here both k and l0 are linearly increasing with T
k = 1 + lambda * T
l0 = 1 + lambda * T"""
spring = SpringT(k = lambda T: 2 - 1.99 * T, l0 = lambda T: 1 + T,
                 dkdt = lambda T: -1.99, dl0dt = lambda T: 1,
                 d2kdt2 = lambda T: 0, d2l0dt2= lambda T: 0)

# setup problem
structure = Problem(spring, ixp=ixp, qp=qp)

# setup solver
solver = IterativeSolver(structure)

# initial point
p0 = Point(qp=np.array([0, 0, 1, 0, 0]))

# solve for equilibrium given initial point
dp0 = solver([p0])[0]

# print("Given L0 = {}, x_1 has to change from {} by {} to {} for equilibrium.".format(spring.nlf.l0, p0.qf[0], dp0.qf[0], p0.qf[0] + dp0.qf[0]))
# setup stepper
steppah = IncrementalSolver(solver)

# solve problem from equilibrium point
solution = steppah(p0 + dp0)[0]

# plot temperature and entropy (flux)
plt.plot([i.y for i in solution], [i.qp[4] for i in solution], 'ko-')
plt.plot([i.y for i in solution], [i.fp[4] for i in solution], 'ko--')

# plot reaction forces on nodes
plt.plot([i.y for i in solution], [i.fp[0] for i in solution], 'ro-')
plt.plot([i.y for i in solution], [i.fp[2] for i in solution], 'ro--')

plt.show()





