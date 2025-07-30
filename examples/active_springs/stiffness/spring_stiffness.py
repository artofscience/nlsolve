from matplotlib import pyplot as plt
import numpy as np

from constraints import GeneralizedArcLength, ArcLength, NewtonRaphson
from core import IncrementalSolver, IterativeSolver
from utils import Problem, Point
from criteria import residual_norm, CriterionY, CriterionXH, CriterionYH
from math import sqrt
from controllers import Adaptive
from operator import le, lt, gt

from spring import SpringK


"""
Observation of disconnected branches when controlling stiffness as DOF.

"""

ixf = [3]
ixp = [0, 1, 2, 4]

ff = np.zeros(1)
qp = np.zeros(len(ixp))
qp[-1] = -3

spring = Problem(SpringK(l0=sqrt(2)), ixf, ixp, ff, qp)

criteria_1 = CriterionY(lambda x: abs(x), le, 1e-2)
criteria_2 = residual_norm(1e-3)
criteria = criteria_1 & criteria_2


controller = Adaptive(0.1, max=0.1, incr=1.5, decr=0.2, min=0.00001)

p0 = Point(q = np.array([0, 0, 1, 1, 3]), f = np.array([0, 0, 0, -0.1, 0]))

solver_init = IterativeSolver(spring, NewtonRaphson(), criteria)
dp0 = solver_init([p0])[0]
p0 = p0 + dp0

solver = IterativeSolver(spring, GeneralizedArcLength(alpha=0.00001, beta=10), criteria)

stepper = IncrementalSolver(solver, maximum_increments=50)

solution, tries = stepper(p0, controller)

plt.plot([i.q[3] for i in solution], [i.f[-1] for i in solution], 'ro-')
plt.plot([i.q[-1] for i in solution], [i.f[-1] for i in solution], 'bo-')


plt.ylim([-3, 3])
plt.show()