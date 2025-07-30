from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from constraints import GeneralizedArcLength, ArcLength, NewtonRaphson
from core import IncrementalSolver, IterativeSolver
from utils import Problem, Point
from criteria import residual_norm
from math import sqrt
from controllers import Adaptive
from copy import deepcopy

from spring import SpringL0K

ixf = [3]
ixp = [0, 1, 2, 4, 5]

spring = Problem(SpringL0K(), ixf, ixp, np.zeros(1), np.array([0, 0, 0, 0, -1.5]))

# Plot colormap

point = Point(np.array([0, 0, 1, 1, sqrt(2), 2]), f = np.array([0, 0, 0, -0.1, 0, 0]))
xy = np.linspace(2, -2, 30)
k = np.linspace(4,-2, 30)
r = np.zeros((len(k), len(xy)), dtype=float)

for ixy, xyname in enumerate(xy):
    for ik, kname in enumerate(k):
        point.q[-1] = kname
        point.q[3] = xyname
        r[ixy, ik] = np.linalg.norm(spring.rf(point))

k, xy = np.meshgrid(k, xy)
plt.contourf(k, xy, r, 100, cmap='jet')
plt.scatter((k**2).flatten(), xy.flatten(), c=r.flatten())
plt.colorbar()


controller = Adaptive(0.01, max=0.05, incr=1.5, decr=0.2, min=0.00001)

p0 = point

solver = IterativeSolver(spring, NewtonRaphson())
dp0 = solver([p0])[0]
p0 = p0 + dp0

stepper = IncrementalSolver(solver)

solution, tries = stepper(p0, controller)

plt.plot([i.q[-1] for i in solution], [i.q[3] for i in solution], 'ro-')

plt.xlim([-2, 2])
plt.ylim([-2, 2])

spring.qpc[-1] = 1.9
p2 = deepcopy(solution[-1])


solver = IterativeSolver(spring, GeneralizedArcLength())
stepper = IncrementalSolver(solver)
solution2, _ = stepper(p2, controller)
plt.plot([i.q[-1] for i in solution2], [i.q[3] for i in solution2], 'ko-')


plt.show()





