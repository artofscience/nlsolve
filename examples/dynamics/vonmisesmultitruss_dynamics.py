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
from matplotlib.pyplot import figure


def main(a: int = 1, max_load: float = 1.0, v0: float = 1.0):
    nlf = StructureFromSpringableModelFile("../springable_curves/csv_files/von_mises_multi_truss.csv")

    if a == 1:
        problem = Problem(nlf, ixf=[2, 4], ixp=[0, 1, 3, 5], ff=np.array([0.0, 1.0]), qp=np.array([0, 0, 0, 0]))
    else:
        problem = Problem(nlf, ixf=[2], ixp=[0, 1, 3, 4, 5], ff=np.array([0.0]), qp=np.array([0, 0, 0, 1, 0]))

    solver = IterativeSolver(problem)

    load = termination_default(max_load)
    criterion = load | EigenvalueChangeTermination()

    stepper = IncrementalSolver(solver, terminated=criterion, reset=False)

    while not load.exceed: stepper()

    pc = stepper.history[0].solutions[-1]  # get first critical point
    dynsolver = DynamicsSolver(problem)  # setup dynamics solver

    dynsolver(pc, m=1.0, v0=v0)

    dynsolver(dynsolver.load_based_offset(pc))  # run solver using first order ODE with load-based offset

    # POST-PROC

    plotter = Plotter()
    for out in stepper.history:
        plotter(out.solutions, 4, 4)
        plotter(out.solutions, 2, 4)

    for out in dynsolver.history:
        plotter(out, 4, 4)
        plotter(out, 2, 4)


if __name__ == '__main__':
    main(1)

    figure()
    main(0, 5.0, -1.0)

    plt.show()
