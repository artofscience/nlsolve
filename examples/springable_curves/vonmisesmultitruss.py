import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(os.path.abspath(os.path.join(script_dir, '../..')))

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

from structure_from_springable import StructureFromSpringableModelFile
from utils import Problem, Plotter
from core import IterativeSolver, IncrementalSolver
from criteria import EigenvalueChangeTermination, termination_default

def main(a):
    nlf = StructureFromSpringableModelFile("csv_files/von_mises_multi_truss.csv")

    if a == 1:
        problem = Problem(nlf, ixp=nlf.get_default_ixp(), ixf=nlf.get_default_ixf(), ff=nlf.get_default_ff(), qp=nlf.get_default_qp())
        max_load = 1.0
    else:
        problem = Problem(nlf, ixf=[2], ixp=[0, 1, 3, 4, 5], ff=np.array([0.0]), qp=np.array([0, 0, 0, 1, 0]))
        max_load = 5.0

    solver = IterativeSolver(problem)

    load = termination_default(max_load)
    criterion = load | EigenvalueChangeTermination()

    stepper = IncrementalSolver(solver, terminated=criterion, reset=False)

    while not load.exceed: stepper()

    plotter = Plotter()
    for _, step in enumerate(stepper.history):
        plotter(step.solutions, 4, 4)

    plt.plot([i.solutions[-1].q[4] for i in stepper.history],
             [i.solutions[-1].f[4] for i in stepper.history], 'mo', markersize=10)




if __name__ == '__main__':
    main(1)

    figure()
    main(0)

    plt.show()