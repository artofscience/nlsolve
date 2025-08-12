import numpy as np
from matplotlib import pyplot as plt

from constraints import NewtonRaphson, GeneralizedArcLength
from core import IncrementalSolver, IterativeSolver
from spring import SpringL0
from utils import Problem, Point
from criteria import termination_default, EigenvalueChangeTermination
from dynamics import DynamicsSolver

# setup horizontal spring problem free in y-movement of second node
# with driven rest length from 0 to 2
spring = Problem(SpringL0(), ixf = [3], ixp = [0, 1, 2, 4],
                 ff = np.zeros(1), qp = np.array([0, 0, 0, 2.0]))

solver = IterativeSolver(spring, NewtonRaphson())
p0 = Point(q=np.array([0, 0, 1, 0, 0]), f=np.zeros(5))
p0 += solver([p0])[0]


load = termination_default()
criterion = load | EigenvalueChangeTermination(margin=1e-4)

solver.constraint = GeneralizedArcLength()
stepper = IncrementalSolver(solver, terminated=criterion, reset=False)
stepper.p0 = p0

while not load.exceed:
    stepper()
    plt.plot([i.q[4] for i in stepper.out.solutions], [i.q[3] for i in stepper.out.solutions], 'ro-')

pc = stepper.history[0].solutions[-1]  # get first critical point
dynsolver = DynamicsSolver(spring)  # setup dynamics solver

# dynamics using velocity
dynsolver(pc, c=0.1, m=1.0, v0=1, tol=1e-6)
plt.plot([i.q[4] for i in dynsolver.history[0]], [i.q[3] for i in dynsolver.history[0]], 'ko-')

stepper2 = IncrementalSolver(solver)
stepper2(dynsolver.history[0][-1], y=stepper.history[0].time[-1])
plt.plot([i.q[4] for i in stepper2.out.solutions], [i.q[3] for i in stepper2.out.solutions], 'mo-')

dynsolver(pc, c=0.1, m=1.0, v0=-1, tol=1e-6)
plt.plot([i.q[4] for i in dynsolver.history[1]], [i.q[3] for i in dynsolver.history[1]], 'ko-')

stepper3 = IncrementalSolver(solver)
stepper3(dynsolver.history[1][-1], y=stepper.history[0].time[-1])
plt.plot([i.q[4] for i in stepper3.out.solutions], [i.q[3] for i in stepper3.out.solutions], 'bo-')


plt.show()
