from matplotlib import pyplot as plt
import numpy as np

from core import IncrementalSolver, IterativeSolver
from utils import Problem, Point
from spring import Spring

""""
Analysis of a simple spring with fixed stiffness k and rest length l0.

x1 is prescribed such that it moves from 1 to 2

"""

# dofs = [x0, y0, x1, y1]
ixp = [0, 1, 2, 3]

# spring parameters
k, l0 = 1.0, 1.0

# setup loading conditions
qp = np.zeros(4)
qp[2] = 1.0

# setup problem
spring = Problem(Spring(k, l0), ixp=ixp, qp=qp)

# setup solver
solver = IterativeSolver(spring)

# initial point
p0 = Point(q=np.array([0, 0, 1, 0]))

# solve for equilibrium given initial point
dp0 = solver([p0])[0]

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

ax1.plot([i.q[2] for i in solution], [i.q[2] for i in solution], 'ko-')
ax2.plot([i.q[2] for i in solution], [i.f[2] for i in solution], 'ro--')


plt.show()





