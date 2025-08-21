from math import pi, sin

import numpy as np
from matplotlib import pyplot as plt

from constraints import GeneralizedArcLength, NewtonRaphson
from controllers import Adaptive
from core import IncrementalSolver, IterativeSolver
from dynamics import DynamicsSolver
from utils import Problem, Plotter
from examples.inclined_truss_snapback import InclinedTrussSnapback
from criteria import EigenvalueChangeTermination, termination_default, residual_norm
from algo import stepper

truss = InclinedTrussSnapback(w=0.1)
problem = Problem(truss, ixf=[0, 1], ff=np.array([0, 0.5]))

steppers = [stepper(problem, default_positive_direction=i) for i in [True, False]]

dynamics_solver1 = DynamicsSolver(problem)
dynamics_solver2 = DynamicsSolver(problem)
dynamics_solver3 = DynamicsSolver(problem)

plotter_statics = Plotter(linestyle='-')
plotter_dynamics = Plotter(linestyle='--', marker='.')

dofs = [1]

for stepper in steppers:
    stepper.controller.max = 0.05
    while not stepper.terminated.left.exceed:
        stepper()
        for j in dofs: plotter_statics(stepper.out.solutions, j,1)

    critical_points = [i.solutions[-1] for i in stepper.history[:-1]]
    for j in dofs: plt.plot([i.q[j] for i in critical_points],
             [i.f[1] for i in critical_points], 'mo', markersize=10)

    for p0 in critical_points:
        dynamics_solver1(dynamics_solver1.load_based_offset(p0, alpha=1))
        dynamics_solver2(dynamics_solver2.load_based_offset(p0, alpha=-1))
        dynamics_solver3(p0, m=1.0, v0=1.0)
        dynamics_solver3(p0, m=1.0, v0=-1.0)



for history in dynamics_solver1.history:
    for j in dofs:
        plotter_dynamics(history.solutions, j, 1)
        plt.plot(history.solutions[-1].q[j],
                 history.solutions[-1].f[1], 'co', markersize=10)

    solver = IterativeSolver(problem, NewtonRaphson(), residual_norm(1e-15))
    p0 = 1.0 * history.solutions[-1]
    p0 = dynamics_solver1.load_based_offset(p0, alpha=-1)
    tries = solver([p0])[3]
    # point = p0 + dp0
    for j in dofs:
        plt.plot([k.q[j] for k in tries], [k.f[1] for k in tries], 'bo:', markersize=10)

for history in dynamics_solver2.history:
    for j in dofs:
        plotter_dynamics(history.solutions, j, 1)
        plt.plot(history.solutions[-1].q[j],
                 history.solutions[-1].f[1], 'co', markersize=10)

    solver = IterativeSolver(problem, NewtonRaphson(), residual_norm(1e-15))
    p0 = 1.0 * history.solutions[-1]
    p0 = dynamics_solver1.load_based_offset(p0, alpha=1)
    tries = solver([p0])[3]
    # point = p0 + dp0
    for j in dofs:
        plt.plot([k.q[j] for k in tries], [k.f[1] for k in tries], 'bo:', markersize=10)

for history in dynamics_solver3.history:
    for j in dofs:
        plotter_dynamics(history.solutions, j, 1)
        plt.plot(history.solutions[-1].q[j],
                 history.solutions[-1].f[1], 'co', markersize=10)

    solver = IterativeSolver(problem, NewtonRaphson(), residual_norm(1e-15))
    p0 = 1.0 * history.solutions[-1]
    tries = solver([p0])[3]
    # point = p0 + dp0
    for j in dofs:
        plt.plot([k.q[j] for k in tries], [k.f[1] for k in tries], 'ro:', markersize=5)

plt.show()