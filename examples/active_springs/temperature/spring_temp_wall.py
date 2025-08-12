from math import sqrt

import numpy as np
from matplotlib import pyplot as plt

from constraints import NewtonRaphson, GeneralizedArcLength
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from spring import SpringT
from utils import Problem, Point

# dofs = [x0, y0, x1, y1, T]
ixf = [3]
ixp = [0, 1, 2, 4]

# setup loading conditions
qp = np.zeros(4)
qp[-1] = 1.0

ff = np.zeros(1)

# setup temperature-sensitive spring
"""Assume here both k and l0 are linearly increasing with T
lambda = [0 ... 1]
T = lambda
k = 1 - 2 * T^2
l0 = sqrt(2) - T"""
spring = SpringT(k=lambda T: 1 - 2 * T ** 2, l0=lambda T: sqrt(2) - T,
                 dkdt=lambda T: -4 * T, dl0dt=lambda T: -1,
                 d2kdt2=lambda T: 0, d2l0dt2=lambda T: 0)

# setup problem
structure = Problem(spring, ixp=ixp, qp=qp, ixf=ixf, ff=ff)

# setup solver
solver = IterativeSolver(structure, NewtonRaphson())

# initial point
p0 = Point(q=np.array([0, 0, 1, 0, 0]), f=np.array([0, 0, 0, -0.05, 0]))

# solve for equilibrium given initial point
dp0 = solver([p0])[0]

# print("Given L0 = {}, x_1 has to change from {} by {} to {} for equilibrium.".format(spring.nlf.l0, p0.qf[0], dp0.qf[0], p0.qf[0] + dp0.qf[0]))
# setup stepper

alpha, beta = 0, 1
solver_arc = IterativeSolver(structure, GeneralizedArcLength(alpha=alpha, beta=beta))
steppah = IncrementalSolver(solver_arc, maximum_increments=100)

controller = Adaptive(value=0.01, min=0.0001, max=0.01, decr=0.1, incr=1.1)
# controller = Controller(0.001)

# solve problem from equilibrium point
solution = steppah(p0 + dp0, controller)[0]

fig, ax1 = plt.subplots(2, 1)

T = np.asarray([i.q[-1] for i in solution])
k = 1 - 2 * T ** 2
l0 = sqrt(2) - T

ax1[0].set_xlabel('T')
ax1[0].plot(T, k, 'r.--')
ax1[0].set_ylabel('Stiffness', color='red')
ax1[0].tick_params(axis='y', labelcolor='red')
# ax1[0].axhline(0, color='r')
# ax1[0].axvline(1/sqrt(2), color='r')

ax2 = ax1[0].twinx()
ax2.plot(T, l0, 'b.--')
ax2.set_ylabel('Rest length', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
# ax2.axhline(1, color='b')
# ax2.axvline(sqrt(2)-1, color='b')

ax1[1].set_xlabel('T')
ax1[1].set_ylabel('Position', color='red')
ax1[1].tick_params(axis='y', labelcolor='red')

ax2 = ax1[1].twinx()
ax2.set_ylabel('Load', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# reaction force on wall
ax2.plot(T, [i.f[2] for i in solution], 'b.--')

# plot displacement
ax1[1].plot(T, [i.q[3] for i in solution], 'r.--')
ax1[1].set_ylim([-3, 3])

plt.show()
