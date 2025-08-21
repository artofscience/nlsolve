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

truss = InclinedTrussSnapback(w=0.5)
problem = Problem(truss, ixf=[0, 1], ixp=[], ff=np.array([0, 0.5]))
controller = Adaptive(max=0.1)

max_load = 1.0

criteria = [termination_default(max_load) | EigenvalueChangeTermination(), termination_default(max_load) | EigenvalueChangeTermination()]
constraints = [GeneralizedArcLength(direction=True), GeneralizedArcLength(direction=False)]
solvers = [IterativeSolver(problem, constraint) for constraint in constraints]
steppers = [IncrementalSolver(solvers[i], controller, terminated=criteria[i], reset=False) for i in [0, 1]]

dynamics_solver = DynamicsSolver(problem)

plotter_statics = Plotter(linestyle='-')
plotter_dynamics = Plotter(linestyle='--', marker='.')

dofs = [1]

for stepper in steppers:
    while not stepper.terminated.left.exceed:
        stepper()
        for j in dofs: plotter_statics(stepper.out.solutions, j,1)

    critical_points = [i.solutions[-1] for i in stepper.history[:-1]]
    for j in dofs: plt.plot([i.q[j] for i in critical_points],
             [i.f[1] for i in critical_points], 'mo', markersize=10)

    for p0 in critical_points:
        # dynamics_solver(dynamics_solver.load_based_offset(p0, alpha=1))
        # dynamics_solver(dynamics_solver.load_based_offset(p0, alpha=-1))
        dynamics_solver(p0, m=1.0, v0=1.0)
        dynamics_solver(p0, m=1.0, v0=-1.0)


for history in dynamics_solver.history:
    for j in dofs:
        plotter_dynamics(history.solutions, j, 1)
        plt.plot(history.solutions[-1].q[j],
                 history.solutions[-1].f[1], 'co', markersize=10)

    solver = IterativeSolver(problem, NewtonRaphson(), residual_norm(1e-15))
    p0 = 1.0 * history.solutions[-1]
    dp0 = solver([p0])[0]
    point = p0 + dp0
    for j in dofs: plt.plot(point.q[j],
             point.f[1], 'yo', markersize=10)




plt.show()