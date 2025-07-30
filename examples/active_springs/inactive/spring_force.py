from matplotlib import pyplot as plt
import numpy as np
from logging import ERROR

from core import IncrementalSolver, IterativeSolver
from utils import Problem, Point
from operator import le, ge
from criteria import CriterionP

from spring import Spring


""""
Analysis of a simple spring with fixed stiffness k and rest length l0.

y1 is free and loaded form magnitude 0.0 to 1
initial point is out of equilibrium (length does not equal restlength)
hence we first ensure it is in equilibrium
given x1 = 1.0 and l0 = sqrt(2) we guess y1 approx 0.9 

Note we use some user-defined criteria to define convergence

"""

# dofs = [x0, y0, x1, y1]
ixp, ixf = [0, 1, 2], [3]

# spring parameters
k, l0 = 1.0, np.sqrt(2)

# setup loading conditions
qp, ff = np.zeros(3), np.ones(1)

# setup problem
spring = Problem(Spring(k, l0), ixf, ixp, ff, qp)

# setup solver
solver = IterativeSolver(spring)


# initial point
p0 = Point(q=np.array([0, 0, 1.0, 0.9]))

# solve for equilibrium given initial point
dp0 = solver([p0])[0]

# get equilibrium point
p0eq = p0 + dp0

print("Given L0 = {}, y_1 has to change from {} by {} to {} for equilibrium.".format(spring.nlf.l0, p0.q[3], dp0.q[3], p0.q[3] + dp0.q[3]))
# setup stepper



steppah = IncrementalSolver(
    solution_method = solver,
    name = "Stepper",
    logging_level = ERROR,
    maximum_increments= 15)


# solve problem from equilibrium point
solution, tries = steppah(p0eq)

fig, ax1 = plt.subplots()

ax1.set_xlabel('lambda')
ax1.set_ylabel('Position', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('Load', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# plot
ax2.plot([i.f[3] for i in solution], [i.f[3] for i in solution], 'ko-')
ax1.plot([i.f[3] for i in solution], [i.q[3] for i in solution], 'ro-')
ax2.plot([i.f[3] for i in solution], [i.f[0] for i in solution], 'bo-')

plt.show()





