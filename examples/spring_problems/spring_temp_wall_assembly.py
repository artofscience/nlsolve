from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
from controllers import Controller, Adaptive
from core import IncrementalSolver, IterativeSolver
from utils import Structure, Point
from constraints import ArcLength, NewtonRaphson, GeneralizedArcLength

from spring_defs import SpringT

class SpringAssembly:
    def __init__(self):
        self.spring1 = SpringT(k = lambda T: 1, l0 = lambda T: sqrt(2),
                 dkdt = lambda T: 1, dl0dt = lambda T:0,
                 d2kdt2 = lambda T: 1, d2l0dt2= lambda T: 0)
        self.spring2 = SpringT(k = lambda T: 0.1 , l0 = lambda T: 3 - 3*T,
                 dkdt = lambda T: 0, dl0dt = lambda T: -3,
                 d2kdt2 = lambda T: 0, d2l0dt2= lambda T: 0)
        self.ix1 = [0, 1, 2, 3, 6]
        self.ix2 = [2, 3, 4, 5, 6]

    def force(self, q):
        f = np.zeros(7, dtype=float)
        f1 = self.spring1.force(q[self.ix1])
        f2 = self.spring2.force(q[self.ix2])
        f[self.ix1] += f1
        f[self.ix2] += f2
        return f

    def jacobian(self, q):
        K = np.zeros((7,7), dtype=float)
        K1 = self.spring1.jacobian(q[self.ix1])
        K2 = self.spring2.jacobian(q[self.ix2])
        K[np.ix_(self.ix1, self.ix1)] += K1
        K[np.ix_(self.ix2, self.ix2)] += K2
        return K


# dofs = [x0, y0, x1, y1, x2, y2, T]
ixf = [3]
ixp = [0, 1, 2, 4, 5, 6]

# setup loading conditions
qp = np.zeros(6)
qp[-1] = 1.0

ff = np.zeros(1)

# setup problem
structure = Structure(SpringAssembly(), ixp=ixp, qp=qp, ixf=ixf, ff=ff)

# setup solver
solver = IterativeSolver(structure, NewtonRaphson())

# initial point
p0 = Point(qp=np.array([0, 0, 1, 1, -2, 0]), qf=np.array([1]))

# solve for equilibrium given initial point
dp0 = solver([p0])[0]

# print("Given L0 = {}, x_1 has to change from {} by {} to {} for equilibrium.".format(spring.nlf.l0, p0.qf[0], dp0.qf[0], p0.qf[0] + dp0.qf[0]))
# setup stepper

alpha, beta = 1, 1
solver_arc = IterativeSolver(structure, GeneralizedArcLength(alpha=alpha, beta=beta))
steppah = IncrementalSolver(solver_arc, maximum_increments=1000)

controller = Adaptive(value=0.001, min=0.00001, max=0.1, decr=0.1, incr=1.5)
# controller = Controller(0.001)

# solve problem from equilibrium point
solution = steppah(p0 + dp0, controller)[0]

fig, ax1 = plt.subplots(1, 1)

T = np.asarray([i.y for i in solution])
l0 = 3 - 3 * T

# plot displacement
plt.plot(T, l0, 'ko--', label='Rest length')
plt.plot(T, [i.qf[0] for i in solution], 'ro--', label='Y-position')
plt.ylim([-2, 3.5])
plt.xlabel('Temperature')
plt.ylabel('Position')
plt.legend()
plt.grid()

np.savetxt('y1.csv', [i.qf[0] for i in solution], delimiter=',')

plt.show()





