from matplotlib import pyplot as plt
import numpy as np

from core import IncrementalSolver, IterativeSolver
from utils import Structure, Point

from spring_defs import Spring

class SpringReduced:
    def __init__(self, spring: Spring):
        self.spring = spring

    def force(self, q: np.ndarray) -> np.ndarray:
        force = self.spring.force(np.array([0.0, 0.0, q[0], 0.0]))
        return np.array([force[2]])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        jacobian = self.spring.jacobian(np.array([0.0, 0.0, q[0], 0.0]))
        return np.array([jacobian[[2], [2]]])

springetje = Spring()
spring = SpringReduced(springetje)

# dofs = [x0, y0, x1, y1]
ixf = [0]

# spring parameters
k, l0 = 1.0, 1.0

# setup loading conditions
ff = np.ones(1)

# setup problem
spring = Structure(SpringReduced(Spring(k, l0)), ixf=ixf, ff=ff)

# setup solver
solver = IterativeSolver(spring)

# initial point
p0 = Point(qf=np.array([2]))

# solve for equilibrium given initial point
dp0 = solver([p0])[0]

print("Given L0 = {}, x_1 has to change from {} by {} to {} for equilibrium.".format(spring.nlf.spring.l0, p0.qf[0], dp0.qf[0], p0.qf[0] + dp0.qf[0]))
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
ax2.plot([i.y for i in solution], [i.ff for i in solution], 'ko--')
ax1.plot([i.y for i in solution], [i.qf for i in solution], 'ro-')

plt.show()





