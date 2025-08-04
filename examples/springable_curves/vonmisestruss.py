# Always run relative to the script's folder
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add two levels up to sys.path for module imports
import sys
sys.path.append(os.path.abspath(os.path.join(script_dir, '../..')))

from matplotlib import pyplot as plt

from structure_from_springable import StructureFromSpringableModelFile
from utils import Problem, Plotter
from core import IterativeSolver, IncrementalSolver
from constraints import GeneralizedArcLength
from controllers import Adaptive


nlf = StructureFromSpringableModelFile("csv_files/von_mises_spring_truss.csv")
problem = Problem(nlf, ixp=nlf.get_default_ixp(), ixf=nlf.get_default_ixf(), ff=nlf.get_default_ff(), qp=nlf.get_default_qp())
solver = IterativeSolver(problem, GeneralizedArcLength())
controller = Adaptive(value=0.3, decr=0.5, incr=1.3, min=0.0001)
stepper = IncrementalSolver(solver, controller)

out = stepper()
Plotter()(out.solutions, 7, 7)



plt.show()
