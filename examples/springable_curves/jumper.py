import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from core import IterativeSolver, IncrementalSolver
from criteria import termination_default, EigenvalueChangeTermination
from structure_from_springable import StructureFromCurve
from utils import Problem, Plotter
from controllers import Adaptive

def main(a: int = 0):
    nlf = StructureFromCurve("csv_files/jumper.csv")
    if a == 0:
        problem = Problem(nlf, ixf=[0, 1], ff=np.array([3, 0]))
    else:
        problem = Problem(nlf, ixp=[0], ixf=[1], ff=np.array([0]), qp=np.array([3]))

    solver = IterativeSolver(problem)

    criterion = termination_default() | EigenvalueChangeTermination()
    controller = Adaptive(value=0.1, decr=0.1, incr=1.5, min=0.00001, max=0.2)

    stepper = IncrementalSolver(solver, controller, terminated=criterion, reset=False)

    plotter = Plotter()
    while not criterion.left.exceed:
        stepper()
        plotter(stepper.out.solutions, 0, 0)


if __name__ == '__main__':
    main(0)

    figure()
    main(1)
    plt.show()

