from math import pi, sin

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from constraints import GeneralizedArcLength
from core import IncrementalSolver, IterativeSolver
from utils import Problem, Plotter
from controllers import Adaptive
from examples.inclined_truss_snapback import InclinedTrussSnapback
from criteria import termination_default, EigenvalueChangeTermination
from scipy.integrate import solve_ivp

def dynamics(t, x, problem, f, m: float = 0.1, alpha: float = 1.0, beta: float = 1.0):
    n = len(x)
    pos = x[:n//2]
    vel = x[n//2:]
    K = problem.nlf.jacobian(pos)
    M = m * np.identity(2)
    C = alpha * M + beta * K
    F = f * problem.ffc
    acc = np.linalg.solve(M, F - C @ vel - K @ pos)
    return np.hstack((vel, acc))

truss = InclinedTrussSnapback()

problem = Problem(truss, ixf=[0, 1], ff=np.array([0, 1.0]))

solver = IterativeSolver(problem)

load = termination_default(1.0)
criterion = load | EigenvalueChangeTermination()

stepper = IncrementalSolver(solver, terminated=criterion, reset=False)
stepper.controller.max = 1.0

while not load.exceed: stepper()

for _, step in enumerate(stepper.history):
    plt.plot([i.q[0] for i in step.solutions], [i for i in step.time], 'ko-')
    plt.plot([i.q[1] for i in step.solutions], [i for i in step.time], 'bo-')

cload = stepper.history[0].time[-1]
cstate = stepper.history[0].solutions[-1].q

sol = solve_ivp(dynamics, [0, 5], np.array([cstate[0], cstate[1], 0, 0]), args=(problem, cload))

plt.plot(sol.y[0], cload * np.ones_like(sol.t), 'mo--')
plt.plot(sol.y[1], cload * np.ones_like(sol.t), 'mo--')

figure()
plt.plot(sol.t, sol.y[0])
plt.plot(sol.t, sol.y[1])


plt.show()




