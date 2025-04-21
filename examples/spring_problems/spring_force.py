from matplotlib import pyplot as plt
import numpy as np

from core import IncrementalSolver, IterativeSolver
from utils import Structure, Point

from spring_defs import Spring

# dofs = [x0, y0, x1, y1]
ixp, ixf = [0, 1, 3], [2]

# spring parameters
k, l0 = 1.0, 1.0

# setup loading conditions
qp, ff = np.zeros(3), np.ones(1)

# setup problem
spring = Structure(Spring(k, l0), ixf, ixp, ff, qp)

# setup solver
solver = IterativeSolver(spring)

# initial point
p0 = Point(qp=np.array([0, 0, 0]), qf=np.array([2]))

# solve for equilibrium given initial point
dp0 = solver([p0])[0]

print("Given L0 = {}, x_1 has to change from {} by {} to {} for equilibrium.".format(spring.nlf.l0, p0.qf[0], dp0.qf[0], p0.qf[0] + dp0.qf[0]))
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
ax2.plot([i.y for i in solution], [i.ff for i in solution], 'ko-')
ax1.plot([i.y for i in solution], [i.qf for i in solution], 'ro-')
ax2.plot([i.y for i in solution], [i.fp[0] for i in solution], 'bo-')

plt.show()





