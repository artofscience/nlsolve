from matplotlib import pyplot as plt
import numpy as np
from constraints import GeneralizedArcLength
from controllers import Controller, Adaptive
from core import IncrementalSolver, IterativeSolver
from utils import Structure, Point
from copy import deepcopy
from spring import SpringL0K
from criteria import residual_norm


# free dofs
ixf = [3]

# prescribed dofs
ixp = [0, 1, 2, 4, 5]

# total force = initial force + lambda * ff
ff = np.array([0]) # so no change in force throughout analysis

# total state = initial state + lambda * qp
qp = np.zeros(len(ixp))
qp[-2] = 2.0 # so change initial restlength by 2.0 m

# build spring model
spring = Structure(SpringL0K(), ixf, ixp, ff, qp)

# setup stepper
solverNR = IterativeSolver(spring, converged=residual_norm(1e-1))
stepperNR = IncrementalSolver(solverNR)

solverARC = IterativeSolver(spring, GeneralizedArcLength(alpha=1.0, beta=0.5), converged=residual_norm(1e-12))
stepperARC = IncrementalSolver(solverARC, maximum_increments=2000)

# initial point, i.e. x0 = y0 = 0, x1 = 1 and L0 = 0, constant load of 1e-3 to force the buckling upwards
point = Point(qp=np.array([0, 0, 1, 0, 1]), ff=np.array([1e-2]))
solution, tries = stepperNR(point + solverNR([point])[0], Controller(0.05)) # solve

# get solution and take the last point to use as first point in subsequent analysis
p2 = deepcopy(solution[-1])
p2.y = 0.0 # reset lambda = 0.0
spring.ff = np.array([-1]) # for second analysis we change the load from 1e-3 to (1e-3 + lambda * -2)
spring.qp[-2] = 0.0 # for second analysis we keep L0 = 2m

adaptive = Adaptive(0.01, max=0.3, incr=1.5, decr=0.3, min=0.000001)
solution2, tries2 = stepperARC(p2, adaptive) # solve for second analysis

# # get solution and take the last point to use as first point in subsequent analysis
p3 = deepcopy(solution2[-1])
p3.y = 0.0 # reset lambda = 0.0
spring.ff = np.array([0.55]) # for second analysis we change the load from 1e-3 to (1e-3 + lambda * -2)
solution3, tries3 = stepperARC(p3, adaptive) # solve for second analysis

# # get solution and take the last point to use as first point in subsequent analysis
p4 = deepcopy(solution3[-1])
p4.y = 0.0 # reset lambda = 0.0
spring.qp[-1] = -.999 # for second analysis we change the load from 1e-3 to (1e-3 + lambda * -2)
solution4, tries4 = stepperARC(p4, controller=Adaptive(0.001, max=0.1, incr=1.5, decr=0.1, min =0.00000001)) # solve for second analysis


"""VISUALS"""

fig, ax1 = plt.subplots()

ax1.set_xlabel('lambda')
ax1.set_ylabel('Position', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('Load', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

aa = [i.y for i in solution]
aa2 = [solution[-1].y + i.y for i in solution2]
aa3 = [solution[-1].y + solution2[-1].y + i.y for i in solution3]
aa4 = [solution[-1].y + solution2[-1].y + solution3[-1].y + i.y for i in solution4]


loading = [aa, aa2, aa3, aa4]
solutions = [solution, solution2, solution3, solution4]

for alpha, beta in enumerate(loading):
    ax1.plot(beta, [i.qf for i in solutions[alpha]], 'ro--')
    ax2.plot(beta, [i.ff for i in solutions[alpha]], 'bo--')
    ax1.plot(beta, [i.qp[-2] for i in solutions[alpha]], 'ko--')
    ax1.plot(beta, [i.qp[-1] for i in solutions[alpha]], 'yo--')
    for b in [0, 1,2, 3, 4]:
        ax2.plot(beta, [i.fp[b] for i in solutions[alpha]], 'go--')





# CHECK REACTION FORCE VALUE
plt.show()