from matplotlib import pyplot as plt
import numpy as np
from logging import DEBUG, INFO, ERROR, WARNING, CRITICAL

from core import IncrementalSolver, IterativeSolver
from utils import Structure, Point
from controllers import Adaptive
from operator import lt, le, ge
from criteria import CriterionX, CriterionXH, CriterionY, CriterionYH, CriterionP


from spring_defs import Spring

# dofs = [x0, y0, x1, y1]
ixp, ixf = [0, 1, 2], [3]

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

# get equilibrium point
p0eq = p0 + dp0

print("Given L0 = {}, x_1 has to change from {} by {} to {} for equilibrium.".format(spring.nlf.l0, p0.qf[0], dp0.qf[0], p0.qf[0] + dp0.qf[0]))
# setup stepper

# Setup termination criterium for the stepper.
# Let's terminate when lambda < -1 OR lambda > 1
# CriterionP works direction on a point object
stepper_c1 = CriterionP(lambda p: p.y, ge, 1.0, name="Stepper_c1")
stepper_c2 = CriterionP(lambda p: p.y, le, -1.0, name="Stepper_c2")
termination_criterium = stepper_c1 | stepper_c2
termination_criterium.logger.name = "stepper_ter_c"

steppah = IncrementalSolver(
    solution_method = solver,
    name = "Stepper",
    logging_level = ERROR,
    maximum_increments= 15,
    terminated= termination_criterium)


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
ax2.plot([i.y for i in solution], [i.ff for i in solution], 'ko-')
ax1.plot([i.y for i in solution], [i.qf for i in solution], 'ro-')
ax2.plot([i.y for i in solution], [i.fp[0] for i in solution], 'bo-')

plt.show()





