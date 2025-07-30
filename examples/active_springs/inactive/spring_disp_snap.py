from matplotlib import pyplot as plt
import numpy as np
from math import sqrt

from core import IncrementalSolver, IterativeSolver
from utils import Problem, Point

from spring import Spring


""""
Analysis of a simple spring with fixed stiffness k and rest length l0.

y1 is prescribed such that it moves from 1 to -1 thereby snaping through instable region.

"""

# dofs = [x0, y0, x1, y1]
ixp = [0, 1, 2, 3]

# spring parameters
k, l0 = 1.0, sqrt(2)

# setup loading conditions
qp = np.zeros(4)
qp[2] = -2.0

# setup problem
spring = Problem(Spring(k, l0), ixp=ixp, qp=qp)

# setup solver
solver = IterativeSolver(spring)

# initial point
p0 = Point(qp=np.array([0, 0, 1, 1]))

# solve for equilibrium given initial point
dp0 = solver([p0])[0]

# print("Given L0 = {}, x_1 has to change from {} by {} to {} for equilibrium.".format(spring.nlf.l0, p0.qf[0], dp0.qf[0], p0.qf[0] + dp0.qf[0]))
# setup stepper
steppah = IncrementalSolver(solver)

# solve problem from equilibrium point
solution = steppah(p0 + dp0)[0]

fig, ax1 = plt.subplots()

ax1.set_xlabel('lambda')
ax1.set_ylabel('Position', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('Load', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# plot
ax1.plot([i.y for i in solution], [i.qp[2] for i in solution], 'ro-')
ax2.plot([i.y for i in solution], [i.fp[2] for i in solution], 'bo--')


plt.show()





