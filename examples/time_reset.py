import numpy as np

from inclined_truss_snapback import InclinedTrussSnapback
from constraints import ArcLength
from core import IncrementalSolver, IterativeSolver
from utils import Problem, Point, plotter
from controllers import Adaptive
from matplotlib import pyplot as plt
from decision_criteria import EigenvalueChangeTermination, LoadTermination


nlf = InclinedTrussSnapback(w=0.5)
problem = Problem(nlf, ixf=[0, 1], ff=np.array([0, 1.0]))

solver = IterativeSolver(problem, ArcLength())
stepper = IncrementalSolver(solver, Adaptive(min=0.00001))

out0 = stepper()
out1 = stepper()

plt.figure(1)
plotter(out0.solutions, 0, 1, 'ko-')
plotter(out1.solutions, 0, 1, 'bo-')

##

stepper = IncrementalSolver(solver, Adaptive(), time_reset=False)

out0 = stepper()
out1 = stepper()

plt.figure(2)
plotter(out0.solutions, 0, 1, 'ko-')
plotter(out1.solutions, 0, 1, 'bo-')

##

stepper = IncrementalSolver(solver, Adaptive())

out0 = stepper(terminated=EigenvalueChangeTermination(margin=0.01))
solver.constraint.direction = False
out1 = stepper(out0.solutions[-1])
solver.constraint.direction = True

plt.figure(3)
plotter(out0.solutions, 0, 1, 'ko-')
plotter(out1.solutions, 0, 1, 'bo-')

print(out0.time)
print(out1.time)

##

stepper = IncrementalSolver(solver, Adaptive(), time_reset=False)

out0 = stepper(terminated=EigenvalueChangeTermination(margin=0.01))
solver.constraint.direction = False
out1 = stepper(out0.solutions[-1])
solver.constraint.direction = True

plt.figure(4)
plotter(out0.solutions, 0, 1, 'ko-')
plotter(out1.solutions, 0, 1, 'bo-')

print(out0.time)
print(out1.time)

##

stepper = IncrementalSolver(solver, Adaptive(), time_reset=False)

out0 = stepper(terminated=EigenvalueChangeTermination(margin=0.01))
solver.constraint.direction = False
out1 = stepper(out0.solutions[-1])
solver.constraint.direction = True

# move one full time step wrt last solution
out2 = stepper(out1.solutions[-1], time_reset=True, terminated=LoadTermination(1.0, 0.001))

# note the time reset is still False!
out3 = stepper(out2.solutions[-1])

plt.figure(5)
plotter(out0.solutions, 0, 1, 'ko-')
plotter(out1.solutions, 0, 1, 'bo-')
plotter(out2.solutions, 0, 1, 'yo-')
plotter(out3.solutions, 0, 1, 'co-')



print(out0.time)
print(out1.time)

##
plt.show()
