import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join(script_dir, '../..')))

from matplotlib import pyplot as plt
import numpy as np

from examples.springable_curves.structure_from_springable import StructureFromSpringableModelFile, StructureFromCurve
from utils import Problem, Plotter
from core import IterativeSolver, IncrementalSolver
from criteria import EigenvalueChangeTermination, termination_default
from dynamics import DynamicsSolver
from constraints import GeneralizedArcLength, NewtonRaphson
from algo import stepper as steppahh
from criteria import residual_norm
from matplotlib.pyplot import figure

def main(a):

    nlf = StructureFromCurve("../springable_curves/csv_files/jumper.csv")
    if a == 0:
        problem = Problem(nlf, ixf=[0, 1], ff=np.array([3, 0]))
    else:
        problem = Problem(nlf, ixp=[0], ixf=[1], ff=np.array([0]), qp=np.array([3]))

    dofs = [0]

    critical_points = []
    dynamic_points = []

    steppers = [steppahh(problem, default_positive_direction=i) for i in [True, False]]

    for stepper in steppers:
        stepper()
        while not stepper.terminated.left.exceed:
            critical_points.append(stepper.out.solutions[-1])
            stepper()

    dynamics_solver = DynamicsSolver(problem)
    for p0 in critical_points:
        dynamics_solver(p0, m=0.5, v0=1.0)
        dynamics_solver(p0, m=0.5, v0=-1.0)

    for history in dynamics_solver.history:
        solver = IterativeSolver(problem, NewtonRaphson(), residual_norm(1e-4))
        p0 = 1.0 * history.solutions[-1]
        dp0 = solver([p0])[0]
        dynamic_points.append(p0 + dp0)

    new_dynamic_points = []
    for point in dynamic_points:
        tmp = []
        for critical_point in critical_points:
            p = point + -1.0 * critical_point
            tol = max(critical_point.norm(), point.norm()) / 100
            tmp.append(p.is_zero(tol))
        if not any(tmp):
            new_dynamic_points.append(point)

    ### PLOTTING
    plotter_statics = Plotter(linestyle='-')
    for stepper in steppers:
        for out in stepper.history:
            for j in dofs: plotter_statics(out.solutions, j, 0)

    plotter_dynamics = Plotter(linestyle='--', marker='.')
    for out in dynamics_solver.history:
        for j in dofs:
            plotter_dynamics(out.solutions, j, 0)
            plt.plot(out.solutions[-1].q[j],
                     out.solutions[-1].f[0], 'co', markersize=10)

    for j in dofs:
        plt.plot([i.q[j] for i in critical_points],
                 [i.f[0] for i in critical_points], 'mo', markersize=10)
        plt.plot([i.q[j] for i in dynamic_points],
                 [i.f[0] for i in dynamic_points], 'yo', markersize=10)
        plt.plot([i.q[j] for i in new_dynamic_points],
                 [i.f[0] for i in new_dynamic_points], 'ro', markersize=10)

if __name__ == '__main__':
    main(0)

    # figure()
    # main(0)

    plt.show()