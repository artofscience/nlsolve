import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join(script_dir, '../..')))

from matplotlib import pyplot as plt
import numpy as np

from examples.springable_curves.structure_from_springable import StructureFromSpringableModelFile
from utils import Problem, Plotter
from core import IterativeSolver, IncrementalSolver
from criteria import EigenvalueChangeTermination, termination_default
from dynamics import DynamicsSolver
from constraints import GeneralizedArcLength, NewtonRaphson
from algo import stepper as steppahh
from criteria import residual_norm
from matplotlib.pyplot import figure

def main(a):
    nlf = StructureFromSpringableModelFile("../springable_curves/csv_files/von_mises_multi_truss.csv")

    if a == 1:
        problem = Problem(nlf, ixp=nlf.get_default_ixp(), ixf=nlf.get_default_ixf(), ff=nlf.get_default_ff(), qp=nlf.get_default_qp())
        max_load = 1.0
    else:
        problem = Problem(nlf, ixf=[2], ixp=[0, 1, 3, 4, 5], ff=np.array([0.0]), qp=np.array([0, 0, 0, 1, 0]))
        max_load = 5.0

    steppers = [steppahh(problem, max_load=max_load, default_positive_direction=i) for i in [True, False]]

    dynamics_solver = DynamicsSolver(problem)

    plotter_statics = Plotter(linestyle='-')
    plotter_dynamics = Plotter(linestyle='--', marker='.')

    dofs = [4]

    for stepper in steppers:
        while not stepper.terminated.left.exceed:
            stepper()
            for j in dofs:
                plotter_statics(stepper.out.solutions, j, 4)

        critical_points = [i.solutions[-1] for i in stepper.history[:-1]]
        for j in dofs: plt.plot([i.q[j] for i in critical_points],
                 [i.f[4] for i in critical_points], 'mo', markersize=10)

        for p0 in critical_points:
            dynamics_solver(p0, m=1.0, v0=1.0)
            dynamics_solver(p0, m=1.0, v0=-1.0)


    for history in dynamics_solver.history:
        for j in dofs:
            plotter_dynamics(history.solutions, j, 4)
            plt.plot(history.solutions[-1].q[j],
                     history.solutions[-1].f[4], 'co', markersize=10)

        solver = IterativeSolver(problem, NewtonRaphson(), residual_norm(1e-4))
        p0 = 1.0 * history.solutions[-1]
        dp0 = solver([p0])[0]
        point = p0 + dp0
        for j in dofs: plt.plot(point.q[j],
                 point.f[4], 'yo', markersize=10)

if __name__ == '__main__':
    main(1)

    figure()
    main(0)

    plt.show()