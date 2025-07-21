import numpy as np
from matplotlib import pyplot as plt

from constraints import GeneralizedArcLength
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from criteria import residual_norm
from decision_criteria import EigenvalueTermination, LoadTermination
from examples.inclined_truss_snapback import InclinedTrussSnapback
from utils import Structure, Point

problem = Structure(InclinedTrussSnapback(), ixf=[0, 1], ff=np.array([0, 0.5]))

solver = IterativeSolver(problem, GeneralizedArcLength(), residual_norm(1e-6))

p0 = Point(qf=np.array([0, 0]), ff=np.array([0, 0]))

controller = Adaptive(0.5, max=0.5, incr=1.2, decr=0.1, min=0.0001)

stepper = IncrementalSolver(solver, p0, controller)

# first solve for load termination
decision = LoadTermination(1.0, 0.01)

# stepper.controller_reset = False
solution = stepper(terminated=decision)[0]
plt.plot([i.qf[0] for i in solution], [i.ff[1] for i in solution], 'ko-')
plt.plot([i.qf[1] for i in solution], [i.ff[1] for i in solution], 'bo-')

# then solve for eigenvalue termination
decision = EigenvalueTermination(-0.2, 0.01)

solution = stepper(terminated=decision)[0]
plt.plot([i.qf[0] for i in solution], [i.ff[1] for i in solution], 'ro-')
plt.plot([i.qf[1] for i in solution], [i.ff[1] for i in solution], 'yo-')

plt.show()
