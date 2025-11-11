import numpy as np
from matplotlib import pyplot as plt

from constraints import NewtonRaphson
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from examples.active_springs.spring import SpringT
from utils import Problem, Point
from sympy import Symbol, exp

T = Symbol('T')
k = 1000 * (2.3 - 2.25 / (1 + exp(-0.31 * (T - 47.06))) + 0.0005 * (T - 40)**2 )
l0 = (12.08 - 5.41 / (1 + exp(-0.208 * (T - 54.5)))) /1000

spring = SpringT(l0, k)

length = spring.l0(20) # restlength at 20 deg

# initial point
# p0 = Point(q=np.array([0, 0, length, 0, 20]))

p0 = Point(q=np.array([0, 0, length, 0, 20]))

# dofs = [x0, y0, x1, y1, T]
ixp, ixf = [0, 1, 3, 4], [2]

# setup loading conditions
qp = np.zeros(4)
qp[3] = 60.0

ff = np.zeros(1)

# setup problem
structure = Problem(spring, ixp=ixp, qp=qp, ixf=ixf, ff=ff)

# setup solver
solver = IterativeSolver(structure, NewtonRaphson())

# solve for equilibrium given initial point
dp0 = solver([p0])[0]

controller = Adaptive(value=1, min=0.0001, max=3, decr=0.1, incr=1.1)

# setup stepper
steppah = IncrementalSolver(solver, controller)

# solve problem from equilibrium point
out = steppah(p0 + dp0)
solution = out.solutions

# plot reaction forces on nodes
plt.plot([i.q[4] for i in solution], [i.q[2] for i in solution], 'ro-')


temp = np.linspace(0, 90, 100)
reaction_force = spring.l0(temp)
plt.plot(temp, reaction_force)

plt.show()
