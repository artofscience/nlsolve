from math import pi, sin

from matplotlib.pyplot import figure
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt

from core import IncrementalSolver, IterativeSolver
from criteria import LoadTermination, EigenvalueChangeTermination, termination_default
from utils import Problem, Plotter
from controllers import Adaptive
from operator import gt, lt
from examples.inclined_truss_snapthrough import InclinedTruss

def dynamics(t, x, nlf, f, c: float = 1.0, m: float = 0.1):
    return [x[1], (f - c * x[1] - nlf.jacobian(x[0])[0] * x[0]) / m]

truss = InclinedTruss(pi / 3)
problem = Problem(truss, ixf=[0], ff=np.array([1]))
solver = IterativeSolver(problem)

load = termination_default(1.0)
criterion = load | EigenvalueChangeTermination()

stepper = IncrementalSolver(solver, terminated=criterion, reset=False)

while not load.exceed: stepper()

plotter = Plotter()
for _, step in enumerate(stepper.history): plotter(step.solutions, 0, 0)


cload = stepper.history[0].time[-1]
cstate = stepper.history[0].solutions[-1].q[0]

sol = solve_ivp(dynamics, [0, 5], np.array([cstate, 0]), args=(truss, cload))

plt.plot(sol.y[0], cload * np.ones_like(sol.t), 'mo--')


figure()
plt.plot(sol.t, sol.y[0])
plt.plot(sol.t, sol.y[1])

plt.show()
