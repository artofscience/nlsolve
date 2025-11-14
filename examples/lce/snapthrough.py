import numpy as np
from contourpy.array import codes_from_points
from matplotlib import pyplot as plt

from constraints import NewtonRaphson, GeneralizedArcLength, GeneralizedArcLengthVarCoeff
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from examples.active_springs.spring import SpringT, Spring
from examples.inclined_truss_snapthrough import InclinedTruss
from utils import Problem, Point
from sympy import Symbol, exp
from math import pi, sin

class LCESnapThroughAssembly:
    def __init__(self):
        T = Symbol('T')
        k = 10 * (2.3 - 2.25 / (1 + exp(-0.31 * (T - 47.06))) + 0.0005 * (T - 40) ** 2)
        l0 = (12.08 - 5.41 / (1 + exp(-0.208 * (T - 54.5)))) /10

        self.lce = SpringT(l0, k)
        self.spring = InclinedTruss(pi/2.5)
        self.ix_lce = [0, 1, 2, 3, 4]
        self.ix_spring = [2]

    def force(self, q):
        f = np.zeros(5, dtype=float)
        f[self.ix_lce] += self.lce.force(q[self.ix_lce])
        f[self.ix_spring] += self.spring.force(q[self.ix_spring])
        return f

    def jacobian(self, q):
        K = np.zeros((5, 5), dtype=float)
        K[np.ix_(self.ix_lce, self.ix_lce)] += self.lce.jacobian(q[self.ix_lce])
        K[np.ix_(self.ix_spring, self.ix_spring)] += self.spring.jacobian(q[self.ix_spring])
        return K


# dofs = [x0, y0, x1, y1, x2, y2, T]
ixf = [2]
ixp = [0, 1, 3, 4]

# setup loading conditions
qp = np.zeros(4)
qp[-1] = 90.0

ff = np.zeros(1)

T0 = 20
nlf = LCESnapThroughAssembly()

# setup problem
structure = Problem(nlf, ixp=ixp, qp=qp, ixf=ixf, ff=ff)

# setup solver
solver = IterativeSolver(structure, NewtonRaphson())

# initial point
dl = nlf.lce.l0(T0)
q0 =np.array([0, 0, dl, 0, 20])
f0 = np.array([0, 0, -1, 0, 0])
initial_state = Point(q0)

# solve for equilibrium given initial point
dp0 = solver([initial_state])[0]
# setup stepper

cf = 50 * np.ones(5)
cq = 10 * np.ones(5)
cq[-1] = 1

constraint = GeneralizedArcLengthVarCoeff(cq[ixf], cq[ixp], cf[ixf], cf[ixp])

solver_arc = IterativeSolver(structure, constraint)
steppah = IncrementalSolver(solver_arc)

controller = Adaptive(value=0.1, min=0.00001, max=5, decr=0.1, incr=1.5)

# solve problem from equilibrium point
steppah(initial_state + dp0, controller)
solution = steppah.out.solutions

T = np.asarray([i.q[-1] for i in solution])

plt.subplot(2, 1, 1)
# plot displacement
plt.plot(T, [i.q[2] for i in solution], 'ko--', label=f'Position internal dof')
plt.xlabel('Temperature')
plt.ylabel('Position')

plt.subplot(2,1,2)
plt.plot(T, [i.f[0] for i in solution], 'ko--', label=f'Reaction force base')
plt.xlabel('Temperature')
plt.ylabel('Force')

# plt.legend(loc="lower left")
plt.show()
